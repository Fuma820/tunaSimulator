using System.Collections.Generic;
using UnityEngine;

public class GeneticBoidAgent : MonoBehaviour
{
    private Rigidbody rBody;

    public float separationWeight = 1f;     // 分離の重み
    public float alignmentWeight = 1f;      // 整列の重み
    public float cohesionWeight = 1f;       // 結合の重み
    public float baseSpeed = 10f;           // 前進速度
    public float separationRadius = 5f;     // 近距離反発半径

    public float detectionRadius = 8f;
    public const int GeneCount = 5;
    public static readonly string[] GeneNames =
    {
        "Separation", "Alignment", "Cohesion", "MoveSpeed", "SeparationRadius"
    };

    public static readonly float[] GeneMins = { 0f, 0f, 0f, 0f, 0f };
    public static readonly float[] GeneMaxs = { 1f, 1f, 1f, 10f, 5f };

    public static float ClampGene(int index, float value)
    {
        return Mathf.Clamp(value, GeneMins[index], GeneMaxs[index]);
    }

    private static float RandomGene(int index)
    {
        return Random.Range(GeneMins[index], GeneMaxs[index]);
    }

    public float maxRotationSpeed = 2f;

    public LayerMask agentLayer;
    public LayerMask obstacleLayer;

    [SerializeField]
    [Min(8)] private int detectionBufferCapacity = 128;

    [SerializeField, Min(1)] private int maxAgentsConsidered = 10;

    private readonly List<GeneticBoidAgent> nearbyAgents = new();
    private readonly List<Collider> nearbyObstacles = new();
    private readonly List<GeneticBoidAgent> nearestAgentsBuffer = new();
    private Collider[] agentColliderBuffer;
    private Collider[] obstacleColliderBuffer;
    private bool agentBufferOverflowLogged;
    private bool obstacleBufferOverflowLogged;

    public float fitness = 0f;
    
    private float totalGroupCohesion = 0f;
    private float totalSmoothness = 0f;
    private float totalSpeed = 0f;
    private int evaluationSteps = 0;

    private void Awake()
    {
        EnsureRigidbody();
    }

    void Start()
    {
        EnsureRigidbody();
        ResetFitness();
    }

    private void EnsureRigidbody()
    {
        if (rBody == null)
        {
            rBody = GetComponent<Rigidbody>();
        }
    }

    public void RandomizeGenes()
    {
        var genes = GenerateRandomGenes();
        ApplyGeneValues(genes);
    }

    /// <summary>
    /// 遺伝子を設定
    /// </summary>
    public void SetGenes(float[] genes)
    {
        if (genes == null || genes.Length != GeneCount) return;

        ApplyGeneValues(genes);
    }

    /// <summary>
    /// 遺伝子を取得
    /// </summary>
    public float[] GetGenes()
    {
        return new float[]
        {
            separationWeight,
            alignmentWeight,
            cohesionWeight,
            baseSpeed,
            separationRadius
        };
    }

    public static float[] GenerateRandomGenes()
    {
        var genes = new float[GeneCount];
        for (int i = 0; i < GeneCount; i++)
        {
            genes[i] = RandomGene(i);
        }
        return genes;
    }

    private void ApplyGeneValues(float[] genes)
    {
        separationWeight = ClampGene(0, genes[0]);
        alignmentWeight = ClampGene(1, genes[1]);
        cohesionWeight = ClampGene(2, genes[2]);
        baseSpeed = ClampGene(3, genes[3]);
        separationRadius = ClampGene(4, genes[4]);
    }

    void FixedUpdate()
    {
        DetectNearbyObjects();

        Vector3 agentRepulsion = CalculateSeparation();
        Vector3 obstacleRepulsion = CalculateObstacleAvoidance();
        Vector3 separation = (agentRepulsion + obstacleRepulsion) * separationWeight;
        Vector3 alignment = CalculateAlignment() * alignmentWeight;
        Vector3 cohesion = CalculateCohesion() * cohesionWeight;

        Vector3 targetDirection = transform.forward;
        if (nearbyAgents.Count > 0 || nearbyObstacles.Count > 0)
        {
            targetDirection = separation + alignment + cohesion;
        }

        if (targetDirection.sqrMagnitude > 0.0001f)
        {
            targetDirection.Normalize();
            Quaternion targetRotation = Quaternion.LookRotation(targetDirection, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, maxRotationSpeed * Time.fixedDeltaTime);
        } else {
            targetDirection = transform.forward;
        }

        rBody.velocity = transform.forward * baseSpeed;
    }

    private void DetectNearbyObjects()
    {
        nearbyAgents.Clear();
        nearbyObstacles.Clear();
        EnsureDetectionBuffers();

        int agentHits = Physics.OverlapSphereNonAlloc(transform.position, detectionRadius, agentColliderBuffer, agentLayer, QueryTriggerInteraction.Ignore);
        if (agentHits >= agentColliderBuffer.Length && !agentBufferOverflowLogged)
        {
           agentBufferOverflowLogged = true;
        }

        for (int i = 0; i < agentHits && i < agentColliderBuffer.Length; i++)
        {
            Collider col = agentColliderBuffer[i];
            if (col == null || col.gameObject == gameObject)
            {
                continue;
            }

            if (col.TryGetComponent(out GeneticBoidAgent agent) && agent != this)
            {
                nearbyAgents.Add(agent);
            }
        }

        int obstacleHits = Physics.OverlapSphereNonAlloc(transform.position, detectionRadius, obstacleColliderBuffer, obstacleLayer, QueryTriggerInteraction.Ignore);
        if (obstacleHits >= obstacleColliderBuffer.Length && !obstacleBufferOverflowLogged)
        {
           obstacleBufferOverflowLogged = true;
        }

        for (int i = 0; i < obstacleHits && i < obstacleColliderBuffer.Length; i++)
        {
            Collider col = obstacleColliderBuffer[i];
            if (col != null)
            {
                nearbyObstacles.Add(col);
            }
        }
    }

    /// <summary>
    /// 分離：近くのエージェントから離れる
    /// </summary>
    private Vector3 CalculateSeparation()
    {
        Vector3 accumulated = Vector3.zero;
        int contributingCount = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            if (agent == null) continue;

            Vector3 offset = transform.position - agent.transform.position;
            if (TryBuildRepulsion(offset, offset, out Vector3 contribution))
            {
                accumulated += contribution;
                contributingCount++;
            }
        }

        return FinalizeRepulsion(accumulated, contributingCount);
    }

    /// <summary>
    /// 整列：近くのエージェントと速度を合わせる
    /// </summary>
    private Vector3 CalculateAlignment()
    {
        if (rBody == null)
        {
            return Vector3.zero;
        }

        Vector3 avgVelocity = Vector3.zero;
        int count = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            if (agent == null || agent.rBody == null) continue;

            avgVelocity += agent.rBody.velocity;
            count++;
        }

        if (count > 0)
        {
            avgVelocity /= count;
            Vector3 desired = avgVelocity - rBody.velocity;
            return desired.normalized;
        }

        return Vector3.zero;
    }

    private List<GeneticBoidAgent> GetNearestAgents()
    {
        nearestAgentsBuffer.Clear();

        if (nearbyAgents.Count == 0)
        {
            return nearestAgentsBuffer;
        }

        for (int i = 0; i < nearbyAgents.Count; i++)
        {
            GeneticBoidAgent agent = nearbyAgents[i];
            if (agent == null || agent == this)
            {
                continue;
            }
            nearestAgentsBuffer.Add(agent);
        }

        if (nearestAgentsBuffer.Count <= 1)
        {
            return nearestAgentsBuffer;
        }

        int targetCount = Mathf.Min(maxAgentsConsidered, nearestAgentsBuffer.Count);
        for (int i = 0; i < targetCount; i++)
        {
            int bestIndex = i;
            float bestDistance = DistanceSquaredTo(nearestAgentsBuffer[i]);

            for (int j = i + 1; j < nearestAgentsBuffer.Count; j++)
            {
                float candidateDistance = DistanceSquaredTo(nearestAgentsBuffer[j]);
                if (candidateDistance < bestDistance)
                {
                    bestDistance = candidateDistance;
                    bestIndex = j;
                }
            }

            if (bestIndex != i)
            {
                (nearestAgentsBuffer[i], nearestAgentsBuffer[bestIndex]) = (nearestAgentsBuffer[bestIndex], nearestAgentsBuffer[i]);
            }
        }

        if (nearestAgentsBuffer.Count > maxAgentsConsidered)
        {
            nearestAgentsBuffer.RemoveRange(maxAgentsConsidered, nearestAgentsBuffer.Count - maxAgentsConsidered);
        }

        return nearestAgentsBuffer;
    }

    /// <summary>
    /// 結合：群れの中心に向かう
    /// </summary>
    private Vector3 CalculateCohesion()
    {
        Vector3 centerOfMass = Vector3.zero;
        int count = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            float distance = Vector3.Distance(transform.position, agent.transform.position);
            if (distance > separationRadius)
            {
                centerOfMass += agent.transform.position;
                count++;
            }
        }

        if (count > 0)
        {
            centerOfMass /= count;
            Vector3 towardsCenter = centerOfMass - transform.position;
            return towardsCenter.normalized;
        }

        return Vector3.zero;
    }

    private Vector3 CalculateObstacleAvoidance()
    {
        Vector3 accumulated = Vector3.zero;
        int contributingCount = 0;

        foreach (Collider obstacle in nearbyObstacles)
        {
            if (obstacle == null)
            {
                continue;
            }

            Vector3 closestPoint;
            if (obstacle is MeshCollider meshCollider && !meshCollider.convex)
            {
                closestPoint = obstacle.bounds.ClosestPoint(transform.position);
            }
            else
            {
                closestPoint = obstacle.ClosestPoint(transform.position);
            }

            Vector3 offset = transform.position - closestPoint;
            Vector3 fallbackDirection = transform.position - obstacle.bounds.center;

            if (TryBuildRepulsion(offset, fallbackDirection, out Vector3 contribution))
            {
                accumulated += contribution;
                contributingCount++;
            }
        }

        return FinalizeRepulsion(accumulated, contributingCount);
    }

    private bool TryBuildRepulsion(Vector3 offset, Vector3 fallbackDirection, out Vector3 contribution)
    {
        float distance = offset.magnitude;
        contribution = Vector3.zero;

        if (distance > separationRadius)
        {
            return false;
        }

        Vector3 direction;
        if (distance > Mathf.Epsilon)
        {
            direction = offset / Mathf.Max(distance, 0.0001f);
        } else if (fallbackDirection.sqrMagnitude > 0.0001f) {
            direction = fallbackDirection.normalized;
        } else {
            return false;
        }

        float strength = separationRadius / Mathf.Max(distance, 0.001f);
        contribution = direction * strength;
        return true;
    }

    private static Vector3 FinalizeRepulsion(Vector3 accumulated, int contributingCount)
    {
        if (contributingCount > 0)
        {
            accumulated /= contributingCount;
        }

        return accumulated.sqrMagnitude > 0.0001f ? accumulated.normalized : Vector3.zero;
    }

    public void ResetFitness()
    {
        fitness = 0f;
        totalGroupCohesion = 0f;
        totalSmoothness = 0f;
        totalSpeed = 0f;
        evaluationSteps = 0;
    }

    private void EnsureDetectionBuffers()
    {
        int requiredCapacity = Mathf.Max(maxAgentsConsidered, detectionBufferCapacity);

        if (agentColliderBuffer == null || agentColliderBuffer.Length != requiredCapacity)
        {
            agentColliderBuffer = new Collider[requiredCapacity];
            agentBufferOverflowLogged = false;
        }

        if (obstacleColliderBuffer == null || obstacleColliderBuffer.Length != requiredCapacity)
        {
            obstacleColliderBuffer = new Collider[requiredCapacity];
            obstacleBufferOverflowLogged = false;
        }
    }

    private float DistanceSquaredTo(GeneticBoidAgent agent)
    {
        if (agent == null)
        {
            return float.PositiveInfinity;
        }

        Vector3 offset = agent.transform.position - transform.position;
        return offset.sqrMagnitude;
    }

    public void ResetTransform(Vector3 worldPosition, Quaternion worldRotation)
    {
        EnsureRigidbody();
        rBody.angularVelocity = Vector3.zero;
        rBody.velocity = Vector3.zero;
        transform.SetPositionAndRotation(worldPosition, worldRotation);
    }

    public void ResetPosition()
    {
        EnsureRigidbody();
        rBody.angularVelocity = Vector3.zero;
        rBody.velocity = Vector3.zero;

        transform.localPosition = new Vector3(
            Random.Range(-10f, 10f),
            Random.Range(0f, 5f),
            Random.Range(-10f, 10f)
        );
        transform.localRotation = Quaternion.Euler(
            Random.Range(-10f, 10f),
            Random.Range(0f, 360f),
            Random.Range(-10f, 10f)
        );
    }
}
