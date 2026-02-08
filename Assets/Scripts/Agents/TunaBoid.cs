using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Boidクラス
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class TunaBoid : Boid
{
    [Header("Separation Weights")]
    [SerializeField] private float obstacleAvoidWeight = 1f;
    [SerializeField, Min(1)] private int maxAgentsConsidered = 10;

    private readonly List<BaseAgent> nearestAgentsBuffer = new();

    /// <summary>
    /// 他のエージェントから距離をとるメソッド
    /// </summary>
    /// <returns>分離ベクトル</returns>
    public override Vector3 Separation()
    {
        Vector3 vector = new(0, 0, 0);
        int innerAgentNum = 0;

        var nearbyAgents = GetNearestAgents();

        foreach (var agent in nearbyAgents)
        {
            if (agent == null) continue;
            // 自分自身がリストに含まれる可能性に備えてスキップ
            if (agent.transform == transform) continue;

            Vector3 toMe = transform.position - agent.transform.position;
            float distance = toMe.magnitude;

            if (distance <= innerRadius && distance > Mathf.Epsilon)
            {
                // ゼロ割れ防止のために距離に下限を設けつつ重み付け
                float strength = innerRadius / Mathf.Max(distance, 0.001f);
                vector += toMe.normalized * strength;
                innerAgentNum++;
            }
        }

        if (innerAgentNum > 0)
        {
            vector /= innerAgentNum;
        }

        return vector.normalized;
    }


    /// <summary>
    /// 視界内のエージェントと向きを合わせるメソッド
    /// </summary>
    /// <returns>整列ベクトル</returns>
    public override Vector3 Align()
    {
        Vector3 vector = new();

        var nearbyAgents = GetNearestAgents();

        foreach (var agent in nearbyAgents)
        {
            vector += agent.GetVelocity;
        }

        // カメラ外に移動しないように上下方向を向かないようにする
        vector.y = 0;

        return vector.normalized;
    }

    /// <summary>
    /// 視界内のエージェントの中心座標を目指すメソッド
    /// </summary>
    /// <returns>結合ベクトル</returns>
    public override Vector3 Cohesion()
    {
        Vector3 totalPosition = new(0, 0, 0);

        var nearbyAgents = GetNearestAgents();

        foreach (var agent in nearbyAgents)
        {
            var targetPosition = agent.gameObject.transform.position;

            if (Vector3.Distance(transform.position, targetPosition) > innerRadius)
            {
                // 基準の範囲内の場合ターゲットとする
                totalPosition += targetPosition - transform.position;
            }
        }

        return totalPosition.normalized;
    }

    private List<BaseAgent> GetNearestAgents()
    {
        nearestAgentsBuffer.Clear();

        if (outerList == null || outerList.Count == 0)
        {
            return nearestAgentsBuffer;
        }

        var orderedAgents = outerList
            .Where(agent => agent != null && agent.transform != transform)
            .OrderBy(agent => (agent.transform.position - transform.position).sqrMagnitude)
            .Take(maxAgentsConsidered);

        nearestAgentsBuffer.AddRange(orderedAgents);
        return nearestAgentsBuffer;
    }

    /// <summary>
    /// エージェント以外のオブジェクトから距離をとるメソッド
    /// </summary>
    /// <returns>回避ベクトル</returns>
    public override Vector3 Avoid()
    {
        Vector3 vector = new(0, 0, 0);
        int innerAgentNum = 0;

        foreach (var obstacle in objectList)
        {
            if (obstacle == null) continue;

            Collider targetCollider = obstacle.GetComponent<Collider>();
            if (targetCollider == null) continue;

            Vector3 closestPointOnTarget;

            // ClosestPoint()が使えるColliderタイプかチェック
            if (targetCollider is BoxCollider || 
                targetCollider is SphereCollider || 
                targetCollider is CapsuleCollider || 
                (targetCollider is MeshCollider meshCollider && meshCollider.convex))
            {
                // 自身の位置に最も近いターゲットのコライダーの表面上の点を取得
                closestPointOnTarget = targetCollider.ClosestPoint(transform.position);
            }
            else
            {
                // TerrainColliderや非凸MeshColliderはBoundsを使用
                closestPointOnTarget = targetCollider.bounds.ClosestPoint(transform.position);
            }

            Vector3 toMe = transform.position - closestPointOnTarget;
            float distance = toMe.magnitude;

            if (distance <= innerRadius)
            {
                // ゼロ距離時はコライダー中心からの押し出し方向を使用
                Vector3 dir = distance > Mathf.Epsilon
                    ? toMe.normalized
                    : (transform.position - targetCollider.bounds.center).normalized;

                float strength = innerRadius / Mathf.Max(distance, 0.001f);
                vector += dir * strength;
                innerAgentNum++;
            }
        }

        // 回避対象が1つ以上あれば平均化
        if (innerAgentNum > 0)
        {
            vector /= innerAgentNum;
        }

        return vector.normalized;
    }

    /// <summary>
    /// エージェントのメインミッションを実行する抽象メソッド
    /// </summary>
    /// <returns></returns>
    public override Vector3 ExecutedMission()
    {
        return ExecuteForwardMission();
    }

    /// <summary>
    /// 前方へ進むミッションメソッド
    /// </summary>
    /// <returns>移動ベクトル</returns>
    private Vector3 ExecuteForwardMission()
    {
        Vector3 targetDirection = transform.forward;

        if (outerList.Count > 0 || objectList.Count > 0)
        {
            Vector3 separationVector = Separation() * separatePower;
            Vector3 alignmentVector = Align() * alignPower;
            Vector3 cohesionVector = Cohesion() * cohesionPower;
            Vector3 obstacleVector = Avoid() * obstacleAvoidWeight;

            targetDirection = separationVector + alignmentVector + cohesionVector + obstacleVector;
            targetDirection += transform.forward * targetDirection.magnitude * destinationPower;
        }

        targetDirection.y = 0;
        if (targetDirection.sqrMagnitude < 0.0001f)
        {
            targetDirection = transform.forward;
            targetDirection.y = 0;
        }

        // 回転して前方をtargetDirectionに向ける（スムーズにしたい場合はSlerpを使う）
        if (targetDirection.sqrMagnitude > 0.0001f)
        {
            targetDirection.Normalize();
            Quaternion targetRotation = Quaternion.LookRotation(targetDirection, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed); // rotationSpeedで回転速度を調整
        }

        // 移動は常にtransform.forward方向
        return transform.forward;
    }

    /// <summary>
    /// 目的地へ進むミッションメソッド
    /// </summary>
    /// <returns>移動ベクトル</returns>
    private Vector3 ExecuteTargetMission()
    {
        Vector3 direction = (destination - transform.position).normalized;
        Vector3 targetDirection = direction;
        Vector3 obstacleVector = Avoid() * obstacleAvoidWeight;

        // 目的地周辺にいる場合は結合と分離のみ
        if (Vector3.Distance(transform.position, destination) < 10)
        {
            targetDirection = Separation() * separatePower + Cohesion() * cohesionPower + obstacleVector;
        }
        else if (outerList.Count > 0 || objectList.Count > 0)
        {
            targetDirection = Separation() * separatePower + Align() * alignPower + Cohesion() * cohesionPower + obstacleVector;
            targetDirection += direction * targetDirection.magnitude * destinationPower;
        }

        targetDirection.y = 0;
        if (targetDirection.magnitude < 0.01f)
        {
            targetDirection = direction;
            targetDirection.y = 0;
        }

        // 回転して前方をtargetDirectionに向ける
        if (targetDirection != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(targetDirection, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed);
        }

        // 移動は常にtransform.forward方向
        return transform.forward;
    }

}
