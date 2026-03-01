using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnAwareCameraController : MonoBehaviour
{
    private float moveDurationSeconds = 3f;
    [SerializeField, Range(0f, 45f)] private float lookDownAngleDegrees = 5f;
    [SerializeField] private VideoRecorder videoRecorder;
    [SerializeField] private GeneticAlgorithmManager geneticAlgorithmManager;
    [SerializeField] private float recordingTailSeconds = 0.5f;
    [SerializeField] private float serverScoreTimeoutSeconds = 180f;
    [SerializeField] private bool prioritizeAgentCenter = true;
    [SerializeField] private float preMoveDelaySeconds = 3f;
    [SerializeField] private float enforcedRecordingDurationSeconds = 5f;
    [SerializeField] private int enforcedRecordingFrameRate = 15;

    private Coroutine initializationRoutine;
    private Coroutine evaluationRoutine;
    private Coroutine waitForScoreRoutine;
    private GeneticAlgorithmManager subscribedGaManager;
    private bool initializationComplete;
    private bool hasAlignedOnce;
    private bool warnedFrameCountMismatch;
    private bool startRequestPending;
    private bool waitingForServerScore;
    private int pendingScoreEpisode = -1;
    private readonly List<GeneticBoidAgent> agentSamplingBuffer = new();

    private void LogStatus(string message)
    {
        Debug.Log($"[SpawnAwareCameraController] {message}");
    }

    private void Awake()
    {
        EnsureVideoRecorderReference();

        AttachGeneticAlgorithmManager(geneticAlgorithmManager ?? FindObjectOfType<GeneticAlgorithmManager>());
    }

    private void EnsureVideoRecorderReference()
    {
        if (videoRecorder == null)
        {
            videoRecorder = GetComponent<VideoRecorder>();
        }

        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
        }

        ApplyEnforcedRecorderSettings();
    }

    private void ApplyEnforcedRecorderSettings()
    {
        if (videoRecorder == null)
        {
            return;
        }

        if (enforcedRecordingDurationSeconds > 0f || enforcedRecordingFrameRate > 0)
        {
            videoRecorder.ConfigureRecordingWindow(enforcedRecordingDurationSeconds, enforcedRecordingFrameRate);
        }
    }

    private void AttachGeneticAlgorithmManager(GeneticAlgorithmManager manager)
    {
        if (subscribedGaManager == manager)
        {
            geneticAlgorithmManager = manager;
            return;
        }

        if (subscribedGaManager != null)
        {
            subscribedGaManager.OnIndividualEvaluationStarted -= HandleIndividualEvaluationStarted;
        }

        subscribedGaManager = manager;
        geneticAlgorithmManager = manager;

        if (subscribedGaManager != null)
        {
            subscribedGaManager.OnIndividualEvaluationStarted += HandleIndividualEvaluationStarted;
        }
    }

    private void DetachGeneticAlgorithmManager()
    {
        if (subscribedGaManager == null)
        {
            return;
        }

        subscribedGaManager.OnIndividualEvaluationStarted -= HandleIndividualEvaluationStarted;
        subscribedGaManager = null;
    }

    private void OnEnable()
    {
        EnsureVideoRecorderReference();

        AttachGeneticAlgorithmManager(geneticAlgorithmManager ?? FindObjectOfType<GeneticAlgorithmManager>());

        if (videoRecorder != null)
        {
            videoRecorder.OnRecordingSegmentCompleted += HandleRecordingSegmentCompleted;
            videoRecorder.OnServerScoreReceived += HandleServerScoreReceived;
        }

        if (!initializationComplete && initializationRoutine == null)
        {
            initializationRoutine = StartCoroutine(SetupAndBeginLoop());
        }
    }

    private void OnDisable()
    {
        if (videoRecorder != null)
        {
            videoRecorder.OnRecordingSegmentCompleted -= HandleRecordingSegmentCompleted;
            videoRecorder.OnServerScoreReceived -= HandleServerScoreReceived;
        }

        DetachGeneticAlgorithmManager();

        if (initializationRoutine != null)
        {
            StopCoroutine(initializationRoutine);
            initializationRoutine = null;
        }

        if (waitForScoreRoutine != null)
        {
            StopCoroutine(waitForScoreRoutine);
            waitForScoreRoutine = null;
        }

        if (evaluationRoutine != null)
        {
            StopCoroutine(evaluationRoutine);
            evaluationRoutine = null;
        }

        startRequestPending = false;
        waitingForServerScore = false;
        pendingScoreEpisode = -1;
        initializationComplete = false;
        hasAlignedOnce = false;
        warnedFrameCountMismatch = false;

    }

    private IEnumerator SetupAndBeginLoop()
    {
        yield return new WaitUntil(IsSpawnContextReady);

        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            initializationComplete = true;
            initializationRoutine = null;
            yield break;
        }

        int configuredFrames = videoRecorder.FramesPerSegment;
        if (configuredFrames > 0 && configuredFrames != 75 && !warnedFrameCountMismatch)
        {
            warnedFrameCountMismatch = true;
        }

        yield return EnsureCameraPosition(true, prioritizeAgentCenter);

        hasAlignedOnce = true;
        initializationComplete = true;
        initializationRoutine = null;
    }

    private void HandleIndividualEvaluationStarted(int generationIndex, int individualIndex)
    {
        if (!isActiveAndEnabled)
        {
            return;
        }

        if (evaluationRoutine != null)
        {
            StopCoroutine(evaluationRoutine);
            evaluationRoutine = null;
        }

        evaluationRoutine = StartCoroutine(BeginEvaluationWorkflow(generationIndex, individualIndex));
    }

    private IEnumerator BeginEvaluationWorkflow(int generationIndex, int individualIndex)
    {
        while (!initializationComplete)
        {
            yield return null;
        }

        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            evaluationRoutine = null;
            yield break;
        }

        if (!warnedFrameCountMismatch)
        {
            int framesPerSegment = videoRecorder.FramesPerSegment;
            if (framesPerSegment > 0 && framesPerSegment != 75)
            {
                warnedFrameCountMismatch = true;
            }
        }

        if (waitForScoreRoutine != null)
        {
            StopCoroutine(waitForScoreRoutine);
            waitForScoreRoutine = null;
        }

        waitingForServerScore = false;
        pendingScoreEpisode = -1;
        startRequestPending = false;

        bool useInstantMove = !hasAlignedOnce;

        if (preMoveDelaySeconds > 0f)
        {
            yield return new WaitForSeconds(preMoveDelaySeconds);
        }

        yield return EnsureCameraPosition(useInstantMove, true);
        hasAlignedOnce = true;

        if (recordingTailSeconds > 0f)
        {
            yield return new WaitForSeconds(recordingTailSeconds);
        }

        yield return StartRecordingWhenReady();
        LogStatus($"Recording requested for generation {generationIndex + 1}, individual {individualIndex + 1}.");

        evaluationRoutine = null;
    }

    private void HandleRecordingSegmentCompleted(int episodeNumber, int framesCaptured)
    {
        if (!isActiveAndEnabled)
        {
            return;
        }

        int expectedFrames = videoRecorder != null ? videoRecorder.FramesPerSegment : 0;
        if (expectedFrames > 0 && framesCaptured < expectedFrames)
        {
            Debug.LogWarning($"[SpawnAwareCameraController] Segment completed with {framesCaptured} frames (expected {expectedFrames}). Waiting for server score before moving.");
        }

        waitingForServerScore = true;
        pendingScoreEpisode = episodeNumber;
        LogStatus($"Recording segment for episode {episodeNumber} finished. Awaiting server score...");

        if (waitForScoreRoutine != null)
        {
            StopCoroutine(waitForScoreRoutine);
            waitForScoreRoutine = null;
        }

        if (serverScoreTimeoutSeconds > 0f)
        {
            waitForScoreRoutine = StartCoroutine(WaitForServerScoreTimeout(serverScoreTimeoutSeconds));
        }
    }

    private void HandleServerScoreReceived(int episodeNumber, float score)
    {
        if (!waitingForServerScore)
        {
            return;
        }

        if (pendingScoreEpisode >= 0 && pendingScoreEpisode != episodeNumber)
        {
            return;
        }

        LogStatus($"Server score {score:F3} received for episode {episodeNumber}. Forwarding to GA and awaiting next evaluation.");
        try
        {
            geneticAlgorithmManager?.HandleServerScoreReceived(episodeNumber, score);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[SpawnAwareCameraController] Failed to forward server score to GA manager: {ex.Message}");
        }

        waitingForServerScore = false;
        pendingScoreEpisode = -1;

        if (waitForScoreRoutine != null)
        {
            StopCoroutine(waitForScoreRoutine);
            waitForScoreRoutine = null;
        }

        if (!isActiveAndEnabled)
        {
            return;
        }
    }

    private IEnumerator WaitForServerScoreTimeout(float timeoutSeconds)
    {
        float elapsed = 0f;
        while (waitingForServerScore && elapsed < timeoutSeconds)
        {
            elapsed += Time.deltaTime;
            yield return null;
        }

        waitForScoreRoutine = null;

        if (!waitingForServerScore)
        {
            yield break;
        }

        waitingForServerScore = false;
        int timedOutEpisode = pendingScoreEpisode;
        pendingScoreEpisode = -1;
        yield break;
    }

    private IEnumerator EnsureCameraPosition(bool instantMove, bool requireAgentAverage)
    {
        if (!IsSpawnContextReady())
        {
            yield break;
        }

        Vector3 averageWorld;
        while (!TryGetAverage(out averageWorld, requireAgentAverage))
        {
            yield return null;
        }

        yield return MoveCameraTo(averageWorld, instantMove);
    }

    private IEnumerator MoveCameraTo(Vector3 averageWorld, bool instantMove)
    {
        Vector3 targetPosition = averageWorld;

        float duration = instantMove ? 0f : Mathf.Max(0.01f, moveDurationSeconds);
        if (duration <= 0f)
        {
            transform.position = targetPosition;
            PointCameraAt(averageWorld);
            yield break;
        }

        Vector3 initialPosition = transform.position;
        float elapsed = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            transform.position = Vector3.Lerp(initialPosition, targetPosition, Mathf.SmoothStep(0f, 1f, t));
            PointCameraAt(averageWorld);
            yield return null;
        }

        transform.position = targetPosition;
        PointCameraAt(averageWorld);
    }

    private IEnumerator StartRecordingWhenReady()
    {
        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            yield break;
        }

        if (waitingForServerScore)
        {
            yield break;
        }

        if (startRequestPending)
        {
            yield break;
        }

        startRequestPending = true;

        while (videoRecorder != null && (videoRecorder.IsBusy || waitingForServerScore))
        {
            yield return null;
        }

        if (videoRecorder == null)
        {
            startRequestPending = false;
            yield break;
        }

        if (waitingForServerScore)
        {
            startRequestPending = false;
            yield break;
        }

        videoRecorder.StartRecording();

        yield return null;

        startRequestPending = false;
    }

    private bool TryGetAverage(out Vector3 averageWorld, bool requireAgentAverage)
    {
        if (prioritizeAgentCenter || requireAgentAverage)
        {
            if (TryGetAgentAverage(out averageWorld))
            {
                return true;
            }

            if (requireAgentAverage)
            {
                return false;
            }
        }

        return TryGetManagerAverage(out averageWorld);
    }

    private bool TryGetManagerAverage(out Vector3 averageWorld)
    {
        averageWorld = Vector3.zero;
        if (geneticAlgorithmManager == null)
        {
            return false;
        }

        Transform managerTransform = geneticAlgorithmManager.transform;
        int childCount = managerTransform.childCount;
        if (childCount > 0)
        {
            Vector3 sum = Vector3.zero;
            int usedCount = 0;
            foreach (Transform child in managerTransform)
            {
                sum += child.position;
                usedCount++;
            }

            if (usedCount > 0)
            {
                averageWorld = sum / usedCount;
                return true;
            }
        }

        averageWorld = managerTransform.position;
        return false;
    }

    private bool IsSpawnContextReady()
    {
        if (geneticAlgorithmManager == null)
        {
            AttachGeneticAlgorithmManager(FindObjectOfType<GeneticAlgorithmManager>());
        }

        if (geneticAlgorithmManager == null)
        {
            return false;
        }

        var activeAgents = geneticAlgorithmManager.ActiveAgents;
        if (activeAgents != null)
        {
            for (int i = 0; i < activeAgents.Count; i++)
            {
                if (activeAgents[i] != null)
                {
                    return true;
                }
            }
        }

        return geneticAlgorithmManager.transform.childCount > 0;
    }

    private bool TryGetAgentAverage(out Vector3 averageWorld)
    {
        averageWorld = Vector3.zero;

        agentSamplingBuffer.Clear();

        if (geneticAlgorithmManager != null)
        {
            foreach (var agent in geneticAlgorithmManager.ActiveAgents)
            {
                if (agent != null && agent.isActiveAndEnabled)
                {
                    agentSamplingBuffer.Add(agent);
                }
            }
        }

        if (agentSamplingBuffer.Count == 0)
        {
            agentSamplingBuffer.AddRange(FindObjectsOfType<GeneticBoidAgent>());
        }

        if (agentSamplingBuffer.Count == 0)
        {
            return false;
        }

        Vector3 sum = Vector3.zero;
        int count = 0;

        for (int i = 0; i < agentSamplingBuffer.Count; i++)
        {
            GeneticBoidAgent agent = agentSamplingBuffer[i];
            if (agent == null || !agent.isActiveAndEnabled)
            {
                continue;
            }

            sum += agent.transform.position;
            count++;
        }

        if (count == 0)
        {
            return false;
        }

        averageWorld = sum / count;
        return true;
    }

    private void PointCameraAt(Vector3 focusPoint)
    {
        Vector3 baseVector = focusPoint - transform.position;
        Vector3 adjustedFocus = focusPoint;

        if (lookDownAngleDegrees > 0f)
        {
            float horizontalDistance = new Vector2(baseVector.x, baseVector.z).magnitude;
            float offset = Mathf.Tan(Mathf.Deg2Rad * lookDownAngleDegrees) * horizontalDistance;
            adjustedFocus -= Vector3.up * offset;
        }

        Vector3 toFocus = adjustedFocus - transform.position;
        if (toFocus.sqrMagnitude < 0.0001f)
        {
            if (geneticAlgorithmManager != null)
            {
                Vector3 managerCenter = geneticAlgorithmManager.transform.position;
                toFocus = managerCenter - transform.position;
            }

            if (toFocus.sqrMagnitude < 0.0001f)
            {
                return;
            }
        }

        Quaternion lookRotation = Quaternion.LookRotation(toFocus, Vector3.up);
        transform.rotation = lookRotation;
    }
}
