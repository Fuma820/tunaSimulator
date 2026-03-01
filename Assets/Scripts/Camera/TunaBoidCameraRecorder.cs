using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class TunaBoidCameraRecorder : MonoBehaviour
{
    [SerializeField] private bool autoFindBoids = true;
    private readonly List<TunaBoid> trackedBoids = new();

    [SerializeField] private Camera targetCamera;
    [SerializeField] private float positionSmoothTime = 0f;
    [SerializeField] private float lookLerpSpeed = 6f;
    [SerializeField] private float movementIntervalSeconds = 5f;
    [SerializeField] private float movementDurationSeconds = 0f;
    [SerializeField] private bool continuousTracking = false;

    [SerializeField] private bool autoStartRecordingOnPlay = false;
    [SerializeField] private bool loopRecording = true;
    [SerializeField] private VideoRecorder videoRecorder;
    [SerializeField] private float recordingTailSeconds = 0.5f;

    private readonly List<TunaBoid> boidCache = new();
    private Vector3 followVelocity;
    private Coroutine timedRoutine;
    private Coroutine refreshRoutine;
    private bool cameraMidMove;

    private void Awake()
    {
        if (targetCamera == null)
        {
            targetCamera = GetComponent<Camera>();
        }

        if (autoFindBoids)
        {
            RefreshTrackedBoids();
        }

        EnsureVideoRecorderReference();
    }

    private void OnEnable()
    {
        if (refreshRoutine == null && autoFindBoids)
        {
            refreshRoutine = StartCoroutine(AutoRefreshBoidList());
        }

        if (autoStartRecordingOnPlay)
        {
            StartRecording();
        }
    }

    private void OnDisable()
    {
        if (timedRoutine != null)
        {
            StopCoroutine(timedRoutine);
            timedRoutine = null;
        }

        if (refreshRoutine != null)
        {
            StopCoroutine(refreshRoutine);
            refreshRoutine = null;
        }

        EnsureVideoRecorderReference();
        if (videoRecorder != null && videoRecorder.IsRecording)
        {
            videoRecorder.StopRecording();
        }
    }

    private void LateUpdate()
    {
        if (!continuousTracking)
        {
            return;
        }

        if (!TryGetFocusAverage(out Vector3 centroid))
        {
            return;
        }

        MoveCamera(centroid);
    }

    public void RefreshTrackedBoids()
    {
        trackedBoids.RemoveAll(b => b == null);
        if (!autoFindBoids)
        {
            return;
        }

        boidCache.Clear();
        boidCache.AddRange(FindObjectsOfType<TunaBoid>());
        trackedBoids.Clear();
        trackedBoids.AddRange(boidCache);
    }

    public void StartRecording()
    {
        if (timedRoutine != null)
        {
            return;
        }

        EnsureVideoRecorderReference();
        if (videoRecorder == null)
        {
            return;
        }

        timedRoutine = StartCoroutine(PeriodicMoveAndRecordRoutine());
    }

    public void StopRecording()
    {
        if (timedRoutine != null)
        {
            StopCoroutine(timedRoutine);
            timedRoutine = null;
        }

        EnsureVideoRecorderReference();
        if (videoRecorder != null && videoRecorder.IsRecording)
        {
            videoRecorder.StopRecording();
        }
    }

    private IEnumerator AutoRefreshBoidList()
    {
        float refreshInterval = Mathf.Max(0.1f, movementIntervalSeconds);
        var wait = new WaitForSeconds(refreshInterval);
        while (enabled && autoFindBoids)
        {
            RefreshTrackedBoids();
            yield return wait;
        }
    }

    private bool TryGetFocusAverage(out Vector3 centroid)
    {
        if (TryGetBoidAverage(out centroid))
        {
            return true;
        }

        centroid = Vector3.zero;
        return false;
    }

    private bool TryGetBoidAverage(out Vector3 centroid)
    {
        centroid = Vector3.zero;
        if (trackedBoids == null || trackedBoids.Count == 0)
        {
            return false;
        }

        int count = 0;
        for (int i = trackedBoids.Count - 1; i >= 0; i--)
        {
            var boid = trackedBoids[i];
            if (boid == null || !boid.isActiveAndEnabled)
            {
                trackedBoids.RemoveAt(i);
                continue;
            }

            centroid += boid.transform.position;
            count++;
        }

        if (count == 0)
        {
            return false;
        }

        centroid /= count;
        return true;
    }

    private void MoveCamera(Vector3 centroid)
    {
        if (targetCamera == null)
        {
            return;
        }

        (Vector3 desiredPosition, Quaternion targetRotation) = ComputeCameraGoal(centroid, Time.deltaTime);
        transform.position = Vector3.SmoothDamp(transform.position, desiredPosition, ref followVelocity, positionSmoothTime);
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * lookLerpSpeed);
    }

    private IEnumerator PeriodicMoveAndRecordRoutine()
    {
        do
        {
            if (!TryGetFocusAverage(out Vector3 centroid))
            {
                yield return new WaitForSecondsRealtime(1f);
                continue;
            }

            yield return StartCoroutine(MoveCameraOnce(centroid));
            yield return StartCoroutine(CaptureClipOnce());

            float estimatedClipSeconds = EstimateRecorderClipSeconds();
            float extraWait = Mathf.Max(0f, movementIntervalSeconds - estimatedClipSeconds);
            if (extraWait > 0f)
            {
                yield return new WaitForSecondsRealtime(extraWait);
            }
        }
        while (loopRecording && enabled);

        timedRoutine = null;
    }

    private IEnumerator MoveCameraOnce(Vector3 centroid)
    {
        cameraMidMove = true;
        (Vector3 targetPos, Quaternion targetRot) = ComputeCameraGoal(centroid, movementIntervalSeconds);
        Vector3 startPos = transform.position;
        Quaternion startRot = transform.rotation;

        float duration = Mathf.Max(0.01f, movementDurationSeconds);
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            float smoothT = Mathf.SmoothStep(0f, 1f, t);
            transform.position = Vector3.Lerp(startPos, targetPos, smoothT);
            transform.rotation = Quaternion.Slerp(startRot, targetRot, smoothT);
            yield return null;
        }

        transform.position = targetPos;
        transform.rotation = targetRot;
        cameraMidMove = false;
    }

    private (Vector3 position, Quaternion rotation) ComputeCameraGoal(Vector3 centroid, float deltaSeconds)
    {
        Vector3 targetPosition = centroid;
        Quaternion targetRotation = BuildLookRotation(transform.position, centroid);
        return (targetPosition, targetRotation);
    }


    private Quaternion BuildLookRotation(Vector3 cameraPosition, Vector3 focusPoint)
    {
        Vector3 direction = focusPoint - cameraPosition;
        if (direction.sqrMagnitude < 0.0001f)
        {
            return transform.rotation;
        }

        direction.Normalize();
        return Quaternion.LookRotation(direction, Vector3.up);
    }

    private IEnumerator CaptureClipOnce()
    {
        if (targetCamera == null)
        {
            yield break;
        }

        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            yield break;
        }

        if (videoRecorder.recordingCamera == null)
        {
            videoRecorder.recordingCamera = targetCamera;
        }

        while (videoRecorder != null && videoRecorder.IsBusy)
        {
            yield return null;
        }

        if (videoRecorder == null)
        {
            yield break;
        }

        videoRecorder.StartRecording();

        while (videoRecorder != null && (videoRecorder.IsRecording || videoRecorder.IsBusy))
        {
            yield return null;
        }

        if (recordingTailSeconds > 0f)
        {
            yield return new WaitForSecondsRealtime(recordingTailSeconds);
        }
    }

    private float EstimateRecorderClipSeconds()
    {
        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            return 0f;
        }

        float configuredDuration = videoRecorder.RecordingDurationSeconds;
        if (configuredDuration > 0f)
        {
            return configuredDuration;
        }

        int frames = videoRecorder.FramesPerSegment;
        int frameRate = videoRecorder.RecordingFrameRate;
        if (frames > 0 && frameRate > 0)
        {
            return frames / (float)frameRate;
        }

        return 0f;
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

        if (videoRecorder == null)
        {
            return;
        }

        if (targetCamera == null)
        {
            targetCamera = GetComponent<Camera>();
        }

        if (videoRecorder.recordingCamera == null && targetCamera != null)
        {
            videoRecorder.recordingCamera = targetCamera;
        }
    }
}
