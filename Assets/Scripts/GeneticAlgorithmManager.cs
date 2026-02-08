using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class GeneticAlgorithmManager : MonoBehaviour
{
    public int populationSize = 20;

    public float mutationRate = 0.5f;
    public float mutationStrength = 0.1f;
    public float crossoverRate = 0.8f;

    public int maxGenerations = 100;

    public bool autoRun = true;

    [SerializeField] private int recordingsPerIndividual = 5;

    public GameObject agentPrefab;

    public float spawnRange = 10f;

    public float spawnHeight = 5f;

    [Min(1)] public int agentsPerEvaluation = 100;

    public int spawnMaxAttempts = 100;

    public LayerMask spawnCollisionMask = Physics.DefaultRaycastLayers;

    public float spawnFallbackRadius = 1.5f;

    [SerializeField]
    public float fogDensity = 0.01f;
    public float r = 0.15f;
    public float g = 0.4f;
    public float b = 0.55f;

    [SerializeField]
    private VideoRecorder videoRecorder;

    [SerializeField]
    private float serverScoreTimeoutSeconds = 1800f;

    [SerializeField]
    private string serverHealthUrl = "http://127.0.0.1:8000/health";

    [SerializeField]
    private float serverHealthRetryIntervalSeconds = 2f;

    [SerializeField]
    private int serverHealthRequestTimeoutSeconds = 5;

    private readonly List<GeneticBoidAgent> simulationAgents = new();
    public event System.Action<int, int> OnIndividualEvaluationStarted;
    public IReadOnlyList<GeneticBoidAgent> ActiveAgents => simulationAgents;

    private class Individual
    {
        public float[] Genes;
        public float Fitness;
    }

    private readonly List<Individual> individuals = new();

    private int currentGeneration = 0;
    private int currentIndividualIndex = -1;
    private bool generationInProgress = false;
    private bool waitingForServerScore = false;
    private Coroutine serverScoreTimeoutCoroutine;
    private bool serverReady = false;
    private bool pendingEvaluationStart = false;
    private Coroutine serverHealthCoroutine;
    private float bestFitness = 0f;
    private float avgFitness = 0f;
    private float[] bestGenes = new float[GeneticBoidAgent.GeneCount];
    private int samplesCompletedForCurrentIndividual = 0;
    private float bestScoreForCurrentIndividual = 0f;

    private Bounds agentPrefabBounds;
    private bool agentBoundsInitialized = false;
    private string csvFilePath;
    private bool csvHeaderWritten = false;
    private string populationCsvFilePath;
    private bool populationCsvHeaderWritten = false;

    private int SamplesPerIndividual => Mathf.Max(1, recordingsPerIndividual);

    [SerializeField]
    private bool enableCsvLogging = false;

    [SerializeField]
    private string csvFileName = "ga_progress.csv";

    [SerializeField]
    private bool overwriteCsvOnStart = true;

    [SerializeField]
    private string populationCsvFileName = "ga_population.csv";

    [SerializeField]
    private bool resumeFromCsv = false;

    [SerializeField]
    private string resumeCsvFileName = "";

    [SerializeField]
    private int resumeGenerationNumber = 0;

    private void Awake()
    {
        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
        }
    }

    private void Start()
    {
        if (agentPrefab == null)
        {
            return;
        }

        CacheAgentPrefabBounds();
        InitializeSimulationAgents();
        InitializePopulationGenes();
    TryLoadPopulationFromCsv();
        InitializeCsvLogging();

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }

    private void OnDisable()
    {
        CancelServerScoreTimeoutWatch();

        if (serverHealthCoroutine != null)
        {
            StopCoroutine(serverHealthCoroutine);
            serverHealthCoroutine = null;
        }
    }

    private void InitializeSimulationAgents()
    {
        CacheAgentPrefabBounds();

        foreach (var agent in simulationAgents)
        {
            if (agent != null)
            {
                Destroy(agent.gameObject);
            }
        }
        simulationAgents.Clear();

        int desiredCount = Mathf.Max(1, agentsPerEvaluation);

        for (int i = 0; i < desiredCount; i++)
        {
            Quaternion spawnRotation;
            Vector3 spawnPosition;

            if (!TryGenerateSpawnPose(out spawnPosition, out spawnRotation))
            {
                Vector3 fallbackLocal = GenerateRandomSpawnLocalPosition();
                spawnPosition = transform.TransformPoint(fallbackLocal);
                spawnRotation = transform.rotation * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            }

            GameObject agentObj = Instantiate(agentPrefab, spawnPosition, spawnRotation, transform);
            agentObj.name = $"BoidAgent_Active_{i:00}";

            if (agentObj.TryGetComponent(out GeneticBoidAgent agent))
            {
                agent.ResetFitness();
                simulationAgents.Add(agent);
            }
        }
    }

    private void InitializePopulationGenes()
    {
        individuals.Clear();
        for (int i = 0; i < populationSize; i++)
        {
            individuals.Add(new Individual
            {
                Genes = GeneticBoidAgent.GenerateRandomGenes(),
                Fitness = 0f
            });
        }
    }

    private void EnsureSimulationAgentsReady()
    {
        int desiredCount = Mathf.Max(1, agentsPerEvaluation);
        if (simulationAgents.Count != desiredCount || simulationAgents.Any(agent => agent == null))
        {
            InitializeSimulationAgents();
        }
    }

    private void CacheAgentPrefabBounds()
    {
        if (agentBoundsInitialized || agentPrefab == null)
        {
            return;
        }

        var boxColliders = agentPrefab.GetComponentsInChildren<BoxCollider>(true);
        Bounds calculatedBounds = new Bounds(Vector3.zero, Vector3.zero);
        bool hasBounds = false;

        if (boxColliders != null && boxColliders.Length > 0)
        {
            foreach (BoxCollider box in boxColliders)
            {
                if (box == null)
                {
                    continue;
                }

                Transform colTransform = box.transform;
                Vector3 localCenter = box.center;
                Vector3 halfSize = box.size * 0.5f;

                for (int x = -1; x <= 1; x += 2)
                {
                    for (int y = -1; y <= 1; y += 2)
                    {
                        for (int z = -1; z <= 1; z += 2)
                        {
                            Vector3 localCorner = localCenter + Vector3.Scale(halfSize, new Vector3(x, y, z));
                            Vector3 rootSpace = colTransform.TransformPoint(localCorner);

                            if (!hasBounds)
                            {
                                calculatedBounds = new Bounds(rootSpace, Vector3.zero);
                                hasBounds = true;
                            }
                            else
                            {
                                calculatedBounds.Encapsulate(rootSpace);
                            }
                        }
                    }
                }
            }
        }

        if (!hasBounds)
        {
            Collider[] colliders = agentPrefab.GetComponentsInChildren<Collider>(true);
            foreach (Collider collider in colliders)
            {
                if (collider == null)
                {
                    continue;
                }

                if (collider is BoxCollider)
                {
                    continue;
                }

                Transform colTransform = collider.transform;

                switch (collider)
                {
                    case SphereCollider sphere:
                    {
                        Vector3 center = colTransform.TransformPoint(sphere.center);
                        Vector3 lossyScale = ToAbsVector(colTransform.lossyScale);
                        float maxScale = Mathf.Max(lossyScale.x, lossyScale.y, lossyScale.z);
                        float radius = sphere.radius * Mathf.Max(maxScale, 0.0001f);

                        Vector3 min = center - Vector3.one * radius;
                        Vector3 max = center + Vector3.one * radius;

                        if (!hasBounds)
                        {
                            calculatedBounds = new Bounds(center, Vector3.zero);
                            hasBounds = true;
                        }

                        calculatedBounds.Encapsulate(min);
                        calculatedBounds.Encapsulate(max);
                        break;
                    }
                    case CapsuleCollider capsule:
                    {
                        Vector3 center = colTransform.TransformPoint(capsule.center);
                        Vector3 lossyScale = ToAbsVector(colTransform.lossyScale);
                        float radius = capsule.radius * Mathf.Max(lossyScale.x, lossyScale.y, lossyScale.z);
                        float axisScale = capsule.direction switch
                        {
                            0 => lossyScale.x,
                            1 => lossyScale.y,
                            2 => lossyScale.z,
                            _ => lossyScale.y
                        };
                        float halfAxis = Mathf.Max(0f, capsule.height * axisScale * 0.5f - radius);
                        float effectiveRadius = radius + halfAxis;

                        Vector3 min = center - Vector3.one * effectiveRadius;
                        Vector3 max = center + Vector3.one * effectiveRadius;

                        if (!hasBounds)
                        {
                            calculatedBounds = new Bounds(center, Vector3.zero);
                            hasBounds = true;
                        }

                        calculatedBounds.Encapsulate(min);
                        calculatedBounds.Encapsulate(max);
                        break;
                    }
                    case MeshCollider meshCollider when meshCollider.sharedMesh != null:
                    {
                        Mesh sharedMesh = meshCollider.sharedMesh;
                        Vector3[] vertices = sharedMesh.vertices;

                        foreach (Vector3 vertex in vertices)
                        {
                            Vector3 worldVertex = colTransform.TransformPoint(vertex);
                            if (!hasBounds)
                            {
                                calculatedBounds = new Bounds(worldVertex, Vector3.zero);
                                hasBounds = true;
                            }
                            else
                            {
                                calculatedBounds.Encapsulate(worldVertex);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (!hasBounds)
        {
            float radius = Mathf.Max(spawnFallbackRadius, 0.1f);
            calculatedBounds = new Bounds(Vector3.zero, Vector3.one * radius * 2f);
        }

        agentPrefabBounds = calculatedBounds;
        agentBoundsInitialized = true;
    }

    private Vector3 GenerateRandomSpawnLocalPosition()
    {
        float angle = Random.Range(0f, Mathf.PI * 2f);
        float distance = Random.Range(0f, Mathf.Max(spawnRange, 0f));
        float height = spawnHeight > 0f ? Random.Range(0f, spawnHeight) : 0f;

        Vector3 horizontal = new Vector3(Mathf.Cos(angle) * distance, 0f, Mathf.Sin(angle) * distance);
        return new Vector3(horizontal.x, height, horizontal.z);
    }

    private bool TryGenerateSpawnPose(out Vector3 worldPosition, out Quaternion worldRotation, GeneticBoidAgent agentToIgnore = null)
    {
        CacheAgentPrefabBounds();

        int attempts = Mathf.Max(1, spawnMaxAttempts);
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            Vector3 localPosition = GenerateRandomSpawnLocalPosition();
            Quaternion localRotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            worldPosition = transform.TransformPoint(localPosition);
            worldRotation = transform.rotation * localRotation;

            if (!IsSpawnPositionBlocked(worldPosition, worldRotation, agentToIgnore))
            {
                return true;
            }
        }

        worldPosition = default;
        worldRotation = default;
        return false;
    }

    private void ResetAgentTransformsForEvaluation()
    {
        foreach (var agent in simulationAgents)
        {
            if (agent == null)
            {
                continue;
            }

            Vector3 spawnPosition;
            Quaternion spawnRotation;

            if (!TryGenerateSpawnPose(out spawnPosition, out spawnRotation, agent))
            {
                Vector3 fallbackLocal = GenerateRandomSpawnLocalPosition();
                spawnPosition = transform.TransformPoint(fallbackLocal);
                spawnRotation = transform.rotation * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            }

            agent.ResetTransform(spawnPosition, spawnRotation);
        }
    }

    private bool IsSpawnPositionBlocked(Vector3 worldPosition, Quaternion worldRotation, GeneticBoidAgent agentToIgnore)
    {
        if (spawnCollisionMask == 0)
        {
            return false;
        }

        if (!agentBoundsInitialized || agentPrefabBounds.size == Vector3.zero)
        {
            float radius = Mathf.Max(spawnFallbackRadius, 0.1f);
            var hits = Physics.OverlapSphere(worldPosition, radius, spawnCollisionMask, QueryTriggerInteraction.Ignore);
            return hits.Any(hit => !IsIgnoredSpawnCollider(hit, agentToIgnore));
        }

        Vector3 parentScale = ToAbsVector(transform.lossyScale);
        Vector3 scaledExtents = Vector3.Scale(agentPrefabBounds.extents, parentScale);
        Vector3 scaledCenter = Vector3.Scale(agentPrefabBounds.center, parentScale);
        Vector3 worldCenter = worldPosition + worldRotation * scaledCenter;

        var overlaps = Physics.OverlapBox(worldCenter, scaledExtents, worldRotation, spawnCollisionMask, QueryTriggerInteraction.Ignore);
        return overlaps.Any(hit => !IsIgnoredSpawnCollider(hit, agentToIgnore));
    }

    private bool IsIgnoredSpawnCollider(Collider collider, GeneticBoidAgent agentToIgnore)
    {
        if (collider == null)
        {
            return true;
        }

        if (collider is not BoxCollider)
        {
            return true;
        }

        Transform targetTransform = collider.transform;
        if (targetTransform == transform)
        {
            return true;
        }

        if (agentToIgnore != null && targetTransform.IsChildOf(agentToIgnore.transform))
        {
            return true;
        }

        return false;
    }

    private static Vector3 ToAbsVector(Vector3 value)
    {
        return new Vector3(Mathf.Abs(value.x), Mathf.Abs(value.y), Mathf.Abs(value.z));
    }

    public void StartEvaluation()
    {
        QueueEvaluationStart();
    }

    private void QueueEvaluationStart()
    {
        pendingEvaluationStart = true;
        EnsureServerHealthCheckStarted();

        if (!serverReady)
        {
            return;
        }

        TryStartEvaluationWhenReady();
    }

    private void TryStartEvaluationWhenReady()
    {
        if (!pendingEvaluationStart || !serverReady)
        {
            return;
        }

        pendingEvaluationStart = false;
        StartGenerationEvaluation();
    }

    private void EnsureServerHealthCheckStarted()
    {
        if (serverReady || serverHealthCoroutine != null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(serverHealthUrl))
        {
            serverReady = true;
            TryStartEvaluationWhenReady();
            return;
        }

        serverHealthCoroutine = StartCoroutine(ServerHealthCheckRoutine());
    }

    public void StartGenerationEvaluation()
    {
        if (!serverReady)
        {
            pendingEvaluationStart = true;
            EnsureServerHealthCheckStarted();
            return;
        }

        pendingEvaluationStart = false;

        EnsureSimulationAgentsReady();

        if (simulationAgents.Count == 0 || individuals.Count == 0)
        {
            return;
        }

        if (generationInProgress)
        {
            return;
        }

        generationInProgress = true;
        currentIndividualIndex = 0;
        PrepareIndividualEvaluation(currentIndividualIndex);
    }

    private IEnumerator ServerHealthCheckRoutine()
    {
        bool isFirstAttempt = true;
        float retryInterval = Mathf.Max(0.1f, serverHealthRetryIntervalSeconds);
        WaitForSeconds waitInstruction = new WaitForSeconds(retryInterval);

        while (!serverReady)
        {
            if (!isFirstAttempt)
            {
                yield return waitInstruction;
            }
            else
            {
                isFirstAttempt = false;
            }

            using (UnityWebRequest request = UnityWebRequest.Get(serverHealthUrl))
            {
                request.timeout = Mathf.Max(1, serverHealthRequestTimeoutSeconds);
                yield return request.SendWebRequest();

                if (IsUnityWebRequestSuccessful(request))
                {
                    serverReady = true;
                    break;
                }
            }
        }

        serverHealthCoroutine = null;
        TryStartEvaluationWhenReady();
    }

    private static bool IsUnityWebRequestSuccessful(UnityWebRequest request)
    {
        if (request == null)
        {
            return false;
        }

        return request.responseCode >= 200 && request.responseCode < 300;
    }

    private void PrepareIndividualEvaluation(int index)
    {
        if (index < 0 || index >= individuals.Count)
        {
            generationInProgress = false;
            waitingForServerScore = false;
            return;
        }

        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
        }

        samplesCompletedForCurrentIndividual = 0;
        bestScoreForCurrentIndividual = 0f;

        if (!TryStartSampleEvaluation())
        {
            HandleSampleStartFailure("VideoRecorder unavailable");
        }
    }

    private bool TryStartSampleEvaluation()
    {
        if (currentIndividualIndex < 0 || currentIndividualIndex >= individuals.Count)
        {
            return false;
        }

        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
            if (videoRecorder == null)
            {
                return false;
            }
        }

        var genes = individuals[currentIndividualIndex].Genes;

        ResetAgentTransformsForEvaluation();

        // フォグ設定を適用（固定値）
        RenderSettings.fogDensity = fogDensity;
        RenderSettings.fogColor = new Color(r, g, b);

        foreach (var agent in simulationAgents)
        {
            if (agent == null)
            {
                continue;
            }

            agent.SetGenes(genes);
            agent.ResetFitness();
        }

        int sampleNumber = samplesCompletedForCurrentIndividual + 1;
        int totalSamples = SamplesPerIndividual;

        videoRecorder.SetEvaluationContext(currentGeneration, currentIndividualIndex, sampleNumber);

        OnIndividualEvaluationStarted?.Invoke(currentGeneration, currentIndividualIndex);

        waitingForServerScore = true;
        BeginServerScoreTimeoutWatch(currentIndividualIndex);

        return true;
    }

    private void HandleSampleStartFailure(string reason)
    {
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();

        float resolvedFitness;
        if (samplesCompletedForCurrentIndividual > 0)
        {
            resolvedFitness = bestScoreForCurrentIndividual;
        }
        else
        {
            resolvedFitness = CalculateFallbackFitness();
        }

        if (currentIndividualIndex >= 0 && currentIndividualIndex < individuals.Count)
        {
            individuals[currentIndividualIndex].Fitness = resolvedFitness;
        }

        AdvanceToNextIndividual();
    }

    private float CalculateFallbackFitness()
    {
        return 0f;
    }

    public void HandleServerScoreReceived(int episodeNumber, float score)
    {
        if (!generationInProgress || !waitingForServerScore)
        {
            return;
        }

        if (currentIndividualIndex < 0 || currentIndividualIndex >= individuals.Count)
        {
            return;
        }

        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();

        float clampedScore = Mathf.Max(0f, score);
        CompleteSampleEvaluation(clampedScore);
    }

    private void CompleteSampleEvaluation(float sampleScore)
    {
        if (currentIndividualIndex < 0 || currentIndividualIndex >= individuals.Count)
        {
            return;
        }

        sampleScore = Mathf.Max(0f, sampleScore);
        bestScoreForCurrentIndividual = Mathf.Max(bestScoreForCurrentIndividual, sampleScore);
        samplesCompletedForCurrentIndividual++;

        int totalSamples = SamplesPerIndividual;
        if (samplesCompletedForCurrentIndividual < totalSamples)
        {
            if (!TryStartSampleEvaluation())
            {
                HandleSampleStartFailure("VideoRecorder unavailable before next sample");
            }
            return;
        }

        float resolvedScore = bestScoreForCurrentIndividual;
        individuals[currentIndividualIndex].Fitness = resolvedScore;
        AdvanceToNextIndividual();
    }

    private void AdvanceToNextIndividual()
    {
        currentIndividualIndex++;

        if (currentIndividualIndex >= individuals.Count)
        {
            CompleteGeneration();
        }
        else
        {
            PrepareIndividualEvaluation(currentIndividualIndex);
        }
    }

    private void CompleteGeneration()
    {
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();

        CalculateStatistics();
        WriteGenerationCsvRow();
    WritePopulationParametersToCsv(currentGeneration, individuals);

        if (maxGenerations > 0 && currentGeneration + 1 >= maxGenerations)
        {
            return;
        }

    var nextPopulation = CreateNextGeneration();
    individuals.Clear();
    individuals.AddRange(nextPopulation);
    currentGeneration++;

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }

    private List<Individual> CreateNextGeneration()
    {
        var ordered = individuals.OrderByDescending(ind => ind.Fitness).ToList();
        var newPopulation = new List<Individual>();

        while (newPopulation.Count < populationSize)
        {
            var parent1 = SelectParentByRank(ordered);
            var parent2 = SelectParentByRank(ordered);

            var (childGenes1, childGenes2) = PerformCrossover(parent1.Genes, parent2.Genes);

            MutateGenes(childGenes1);
            MutateGenes(childGenes2);

            newPopulation.Add(new Individual { Genes = childGenes1, Fitness = 0f });
            if (newPopulation.Count < populationSize)
            {
                newPopulation.Add(new Individual { Genes = childGenes2, Fitness = 0f });
            }
        }

        return newPopulation;
    }

    private Individual SelectParentByRank(List<Individual> ordered)
    {
        if (ordered == null || ordered.Count == 0)
        {
            return individuals[Random.Range(0, individuals.Count)];
        }

        int count = ordered.Count;
        // Linear rank weights: best individual weight = count, worst = 1
        int totalWeight = (count * (count + 1)) / 2;
        int pickValue = Random.Range(0, totalWeight);
        int cumulative = 0;

        for (int i = 0; i < count; i++)
        {
            int weight = count - i;
            cumulative += weight;
            if (pickValue < cumulative)
            {
                return ordered[i];
            }
        }

        return ordered[^1];
    }

    private (float[], float[]) PerformCrossover(float[] parent1, float[] parent2)
    {
        if (Random.value >= crossoverRate)
        {
            return (CloneGenes(parent1), CloneGenes(parent2));
        }

        return BLXAlphaCrossover(parent1, parent2, 0.5f);
    }

    private static (float[], float[]) BLXAlphaCrossover(float[] parent1, float[] parent2, float alpha)
    {
        int geneCount = GeneticBoidAgent.GeneCount;
        float[] child1 = new float[geneCount];
        float[] child2 = new float[geneCount];

        for (int i = 0; i < geneCount; i++)
        {
            float min = Mathf.Min(parent1[i], parent2[i]);
            float max = Mathf.Max(parent1[i], parent2[i]);
            float range = max - min;

            float lowerBound = min - alpha * range;
            float upperBound = max + alpha * range;

            child1[i] = GeneticBoidAgent.ClampGene(i, Random.Range(lowerBound, upperBound));
            child2[i] = GeneticBoidAgent.ClampGene(i, Random.Range(lowerBound, upperBound));
        }

        return (child1, child2);
    }

    private void MutateGenes(float[] genes)
    {
        if (Random.value > mutationRate)
        {
            return;
        }

        for (int i = 0; i < genes.Length; i++)
        {
            if (Random.value < 0.5f)
            {
                float range = GeneticBoidAgent.GeneMaxs[i] - GeneticBoidAgent.GeneMins[i];
                float mutation = GaussianRandom() * mutationStrength * range;
                genes[i] = GeneticBoidAgent.ClampGene(i, genes[i] + mutation);
            }
        }
    }

    private static float[] CloneGenes(float[] source)
    {
        float[] clone = new float[source.Length];
        source.CopyTo(clone, 0);
        return clone;
    }

    private float GaussianRandom()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
    }

    private void CalculateStatistics()
    {
        if (individuals.Count == 0)
        {
            bestFitness = 0f;
            avgFitness = 0f;
            bestGenes = new float[GeneticBoidAgent.GeneCount];
            return;
        }

        var ordered = individuals.OrderByDescending(ind => ind.Fitness).ToList();
        bestFitness = ordered[0].Fitness;
        avgFitness = individuals.Average(ind => ind.Fitness);
        bestGenes = CloneGenes(ordered[0].Genes);
    }

    private bool TryLoadPopulationFromCsv()
    {
        if (!resumeFromCsv)
        {
            return false;
        }

        string fileName = string.IsNullOrWhiteSpace(resumeCsvFileName) ? populationCsvFileName : resumeCsvFileName.Trim();
        if (string.IsNullOrWhiteSpace(fileName))
        {
            return false;
        }

        string logsDirectory = Path.Combine(Application.dataPath, "..", "Logs");
        string fullPath = Path.GetFullPath(Path.Combine(logsDirectory, fileName));

        if (!File.Exists(fullPath))
        {
            return false;
        }

        try
        {
            var generationRows = new Dictionary<int, List<(int index, float[] genes, float fitness)>>();
            using StreamReader reader = new StreamReader(fullPath, Encoding.UTF8);
            bool headerSkipped = false;

            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                if (!headerSkipped)
                {
                    headerSkipped = true;
                    continue;
                }

                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] columns = line.Split(',');
                if (columns.Length < 2)
                {
                    continue;
                }

                if (!int.TryParse(columns[0].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int generationNumber))
                {
                    continue;
                }

                if (!int.TryParse(columns[1].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int individualNumber))
                {
                    continue;
                }

                float[] genes = new float[GeneticBoidAgent.GeneCount];
                for (int geneIndex = 0; geneIndex < genes.Length; geneIndex++)
                {
                    int valueIndex = 2 + geneIndex;
                    float parsedValue = 0f;

                    if (valueIndex < columns.Length &&
                        float.TryParse(columns[valueIndex].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                    {
                        parsedValue = parsed;
                    }

                    genes[geneIndex] = GeneticBoidAgent.ClampGene(geneIndex, parsedValue);
                }

                // Fitnessカラムを読み取り
                float fitness = 0f;
                int fitnessColumnIndex = 2 + GeneticBoidAgent.GeneCount;
                if (fitnessColumnIndex < columns.Length &&
                    float.TryParse(columns[fitnessColumnIndex].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedFitness))
                {
                    fitness = parsedFitness;
                }

                if (!generationRows.TryGetValue(generationNumber, out var list))
                {
                    list = new List<(int, float[], float)>();
                    generationRows[generationNumber] = list;
                }

                list.Add((individualNumber, genes, fitness));
            }

            if (generationRows.Count == 0)
            {
                return false;
            }

            int latestGeneration = generationRows.Keys.Max();
            int targetGeneration = resumeGenerationNumber > 0 ? resumeGenerationNumber : latestGeneration;
            if (!generationRows.TryGetValue(targetGeneration, out var targetRows))
            {
                targetGeneration = latestGeneration;
                targetRows = generationRows[targetGeneration];
            }

            var orderedRows = targetRows.OrderBy(tuple => tuple.index).ToList();


            var restoredPopulation = new List<Individual>(populationSize);
            for (int i = 0; i < populationSize; i++)
            {
                if (i < orderedRows.Count)
                {
                    restoredPopulation.Add(new Individual
                    {
                        Genes = CloneGenes(orderedRows[i].genes),
                        Fitness = orderedRows[i].fitness
                    });
                }
                else
                {
                    restoredPopulation.Add(new Individual
                    {
                        Genes = GeneticBoidAgent.GenerateRandomGenes(),
                        Fitness = 0f
                    });
                }
            }

            individuals.Clear();
            individuals.AddRange(restoredPopulation);

            // 読み込んだ世代の結果を親として次世代を生成
            currentGeneration = targetGeneration;
            CalculateStatistics();
            var nextPopulation = CreateNextGeneration();
            individuals.Clear();
            individuals.AddRange(nextPopulation);

            if (overwriteCsvOnStart)
            {
                overwriteCsvOnStart = false;
            }
            return true;
        }
        catch (System.Exception)
        {
            return false;
        }
    }

    private void InitializeCsvLogging()
    {
        csvHeaderWritten = false;
        populationCsvHeaderWritten = false;

        if (!enableCsvLogging)
        {
            csvFilePath = null;
            populationCsvFilePath = null;
            return;
        }

        string sanitizedFileName = string.IsNullOrWhiteSpace(csvFileName) ? "ga_progress.csv" : csvFileName.Trim();
        string populationFile = string.IsNullOrWhiteSpace(populationCsvFileName) ? "ga_population.csv" : populationCsvFileName.Trim();
        string logsDirectory = Path.Combine(Application.dataPath, "..", "Logs");

        try
        {
            Directory.CreateDirectory(logsDirectory);
        }
        catch (System.Exception)
        {
            enableCsvLogging = false;
            csvFilePath = null;
            return;
        }

        InitializeCsvFile(logsDirectory, sanitizedFileName, CreateCsvHeader(), ref csvFilePath, ref csvHeaderWritten);
        InitializeCsvFile(logsDirectory, populationFile, CreatePopulationCsvHeader(), ref populationCsvFilePath, ref populationCsvHeaderWritten);
    }

    private string CreateCsvHeader()
    {
        var sb = new StringBuilder();
        sb.Append("Generation,BestFitness,AverageFitness");

        for (int i = 0; i < GeneticBoidAgent.GeneCount; i++)
        {
            sb.Append(',');
            sb.Append(GeneticBoidAgent.GeneNames[i]);
        }

        return sb.ToString();
    }

    private string CreatePopulationCsvHeader()
    {
        var sb = new StringBuilder();
        sb.Append("Generation,Individual");

        for (int i = 0; i < GeneticBoidAgent.GeneCount; i++)
        {
            sb.Append(',');
            sb.Append(GeneticBoidAgent.GeneNames[i]);
        }

        sb.Append(",Fitness");

        return sb.ToString();
    }

    private void InitializeCsvFile(string directory, string fileName, string header, ref string filePath, ref bool headerWritten)
    {
        filePath = Path.GetFullPath(Path.Combine(directory, fileName));
        bool fileExists = File.Exists(filePath);
        bool shouldOverwrite = overwriteCsvOnStart || !fileExists;

        if (!shouldOverwrite && fileExists)
        {
            try
            {
                shouldOverwrite = new FileInfo(filePath).Length == 0;
            }
            catch (System.Exception)
            {
            }
        }

        if (shouldOverwrite)
        {
            try
            {
                using StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8);
                writer.WriteLine(header);
                headerWritten = true;
            }
            catch (System.Exception)
            {
                filePath = null;
                headerWritten = false;
            }
        }
        else
        {
            headerWritten = true;
        }
    }

    private void WriteGenerationCsvRow()
    {
        if (!enableCsvLogging || string.IsNullOrEmpty(csvFilePath))
        {
            return;
        }

        try
        {
            if (!csvHeaderWritten)
            {
                using StreamWriter headerWriter = new StreamWriter(csvFilePath, false, Encoding.UTF8);
                headerWriter.WriteLine(CreateCsvHeader());
                csvHeaderWritten = true;
            }

            int generationNumber = currentGeneration + 1;
            var sb = new StringBuilder();
            sb.Append(generationNumber);
            sb.Append(',');
            sb.Append(bestFitness.ToString("F4", CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(avgFitness.ToString("F4", CultureInfo.InvariantCulture));

            for (int i = 0; i < bestGenes.Length; i++)
            {
                sb.Append(',');
                sb.Append(bestGenes[i].ToString("F4", CultureInfo.InvariantCulture));
            }

            sb.AppendLine();
            File.AppendAllText(csvFilePath, sb.ToString(), Encoding.UTF8);
        }
        catch (System.Exception)
        {
        }
    }

    private void WritePopulationParametersToCsv(int generationIndex, List<Individual> population)
    {
        if (!enableCsvLogging || string.IsNullOrEmpty(populationCsvFilePath))
        {
            return;
        }

        try
        {
            if (!populationCsvHeaderWritten)
            {
                using StreamWriter headerWriter = new StreamWriter(populationCsvFilePath, false, Encoding.UTF8);
                headerWriter.WriteLine(CreatePopulationCsvHeader());
                populationCsvHeaderWritten = true;
            }

            int generationNumber = generationIndex + 1;
            using StreamWriter writer = new StreamWriter(populationCsvFilePath, true, Encoding.UTF8);
            for (int i = 0; i < population.Count; i++)
            {
                var genes = population[i].Genes;
                if (genes == null)
                {
                    genes = new float[GeneticBoidAgent.GeneCount];
                }

                var sb = new StringBuilder();
                sb.Append(generationNumber);
                sb.Append(',');
                sb.Append(i + 1);

                for (int geneIndex = 0; geneIndex < GeneticBoidAgent.GeneCount; geneIndex++)
                {
                    float geneValue = geneIndex < genes.Length ? genes[geneIndex] : 0f;
                    sb.Append(',');
                    sb.Append(geneValue.ToString("F4", CultureInfo.InvariantCulture));
                }

                float fitnessValue = population[i]?.Fitness ?? 0f;
                sb.Append(',');
                sb.Append(fitnessValue.ToString("F4", CultureInfo.InvariantCulture));

                writer.WriteLine(sb.ToString());
            }
        }
        catch (System.Exception)
        {
        }
    }

    public void StopEvaluation()
    {
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();
        pendingEvaluationStart = false;
        autoRun = false;
    }

    public void ResetGA()
    {
        currentGeneration = 0;
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();
        pendingEvaluationStart = false;
        samplesCompletedForCurrentIndividual = 0;
        bestScoreForCurrentIndividual = 0f;
        InitializePopulationGenes();
        InitializeCsvLogging();
        ResetAgentTransformsForEvaluation();

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }

    private void BeginServerScoreTimeoutWatch(int expectedIndividualIndex)
    {
        CancelServerScoreTimeoutWatch();

        if (serverScoreTimeoutSeconds <= 0f)
        {
            return;
        }

        serverScoreTimeoutCoroutine = StartCoroutine(ServerScoreTimeoutRoutine(expectedIndividualIndex));
    }

    private void CancelServerScoreTimeoutWatch()
    {
        if (serverScoreTimeoutCoroutine == null)
        {
            return;
        }

        StopCoroutine(serverScoreTimeoutCoroutine);
        serverScoreTimeoutCoroutine = null;
    }

    private IEnumerator ServerScoreTimeoutRoutine(int expectedIndividualIndex)
    {
        yield return new WaitForSeconds(serverScoreTimeoutSeconds);
        serverScoreTimeoutCoroutine = null;
        HandleServerScoreTimeout(expectedIndividualIndex);
    }

    private void HandleServerScoreTimeout(int expectedIndividualIndex)
    {
        if (!generationInProgress || !waitingForServerScore)
        {
            return;
        }

        if (currentIndividualIndex != expectedIndividualIndex)
        {
            return;
        }

        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();

        float fallbackFitness = CalculateFallbackFitness();
        CompleteSampleEvaluation(fallbackFitness);
    }

}
