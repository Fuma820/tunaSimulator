using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Follows the average position of all active <see cref="TunaBoid"/> agents while
/// optionally orbiting around them, and records short video clips without any
/// server-side integration.
/// </summary>
[RequireComponent(typeof(Camera))]
public class TunaBoidCameraRecorder : MonoBehaviour
{
    [Header("Boid Tracking")]
    [SerializeField] private bool autoFindBoids = true;
    private readonly List<TunaBoid> trackedBoids = new();

    [Header("Camera Motion")]
    [SerializeField] private Camera targetCamera;
    [SerializeField] private float followDistance = 14f;
    [SerializeField, Tooltip("Smooth damp time for camera movement.")]
    private float positionSmoothTime = 0f;
    [SerializeField, Tooltip("Orbit speed around the centroid in degrees/second. Set to 0 to disable orbiting.")]
    private float orbitDegreesPerSecond = 15f;
    [SerializeField, Tooltip("How quickly the camera rotates to face the boid cluster.")]
    private float lookLerpSpeed = 6f;
    [SerializeField, Tooltip("Seconds between camera repositioning events.")]
    private float movementIntervalSeconds = 5f;
    [SerializeField, Tooltip("Duration of each camera move toward the new centroid.")]
    private float movementDurationSeconds = 0f;
    [SerializeField, Tooltip("If true, LateUpdate keeps tracking continuously in addition to the 5-second moves.")]
    private bool continuousTracking = false;
    [SerializeField, Tooltip("Optional object generator used to derive spawn positions for camera placement.")]
    private ObjectGenerator objectGenerator;
    [SerializeField, Tooltip("Maximum distance allowed between camera and boid focus point.")]
    private float maxDistanceToFocus = 20f;
    [SerializeField, Tooltip("Camera pitch angle to look slightly below the focus point.")]
    [Range(0f, 45f)] private float lookDownAngleDegrees = 5f;

    [Header("Recording")]
    [SerializeField] private bool autoStartRecordingOnPlay = false;
    [SerializeField] private bool loopRecording = true;
    [SerializeField, Tooltip("VideoRecorder component responsible for saving the clip. Automatically discovered if left empty.")]
    private VideoRecorder videoRecorder;
    [SerializeField, Tooltip("Extra seconds to wait after the VideoRecorder finishes before the next move/record cycle begins.")]
    private float recordingTailSeconds = 0.5f;

    private readonly List<TunaBoid> boidCache = new();
    private Vector3 followVelocity;
    private Coroutine timedRoutine;
    private Coroutine refreshRoutine;
    private float currentOrbitAngle;
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

    /// <summary>
    /// Manually refreshes the tracked boid list.
    /// </summary>
    public void RefreshTrackedBoids()
    {
        trackedBoids.RemoveAll(b => b == null);
        if (objectGenerator != null)
        {
            PopulateBoidsFromGenerator();
            if (trackedBoids.Count > 0 || !autoFindBoids)
            {
                return;
            }
        }

        if (!autoFindBoids)
        {
            return;
        }

        boidCache.Clear();
        boidCache.AddRange(FindObjectsOfType<TunaBoid>());
        trackedBoids.Clear();
        trackedBoids.AddRange(boidCache);
    }

    /// <summary>
    /// Begins recording a new clip.
    /// </summary>
    public void StartRecording()
    {
        if (timedRoutine != null)
        {
            return;
        }

        EnsureVideoRecorderReference();
        if (videoRecorder == null)
        {
            UnityEngine.Debug.LogWarning("[TunaBoidCameraRecorder] Cannot start recording because no VideoRecorder is assigned.");
            return;
        }

        timedRoutine = StartCoroutine(PeriodicMoveAndRecordRoutine());
    }

    /// <summary>
    /// Stops the current recording loop (if any).
    /// </summary>
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

        if (objectGenerator != null)
        {
            centroid = objectGenerator.transform.position;
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

    private void PopulateBoidsFromGenerator()
    {
        if (objectGenerator == null)
        {
            return;
        }

        trackedBoids.Clear();
        foreach (Transform child in objectGenerator.transform)
        {
            if (child == null)
            {
                continue;
            }

            TunaBoid boid = child.GetComponent<TunaBoid>();
            if (boid == null)
            {
                boid = child.GetComponentInChildren<TunaBoid>();
            }

            if (boid != null)
            {
                trackedBoids.Add(boid);
            }
        }
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
        Vector3 focusPoint = centroid;
        Vector3 orbitCenter = objectGenerator != null
            ? GetGeneratorCenterWorld()
            : focusPoint;
        float orbitRadius = Mathf.Max(0.1f, followDistance);

        Vector3 planarPoint = ComputeNearestPointOnCircle(orbitCenter, orbitRadius, focusPoint, deltaSeconds);
        planarPoint = ClampDistanceToFocus(planarPoint, focusPoint);

        float horizontalDistance = new Vector2(planarPoint.x - focusPoint.x, planarPoint.z - focusPoint.z).magnitude;
        float heightOffset = ComputeHeightOffset(horizontalDistance);
        planarPoint.y = focusPoint.y + heightOffset;

        Quaternion targetRotation = BuildLookRotation(planarPoint, focusPoint);
        return (planarPoint, targetRotation);
    }

    private Vector3 GetGeneratorCenterWorld()
    {
        if (objectGenerator == null)
        {
            return transform.position;
        }

        return objectGenerator.transform.position;
    }

    private Vector3 ComputeNearestPointOnCircle(Vector3 circleCenter, float radius, Vector3 focusPoint, float deltaSeconds)
    {
        Vector3 direction = focusPoint - circleCenter;
        direction.y = 0f;

        if (direction.sqrMagnitude < 0.0001f)
        {
            Vector3 fallback = transform.position - circleCenter;
            fallback.y = 0f;
            if (fallback.sqrMagnitude > 0.0001f)
            {
                direction = fallback;
            }
            else
            {
                currentOrbitAngle += orbitDegreesPerSecond * Mathf.Max(0f, deltaSeconds);
                direction = Quaternion.Euler(0f, currentOrbitAngle, 0f) * Vector3.forward;
            }
        }

        direction.Normalize();
        Vector3 point = circleCenter + direction * radius;
        point.y = circleCenter.y;
        return point;
    }

    private Vector3 ClampDistanceToFocus(Vector3 desiredPosition, Vector3 focusPoint)
    {
        if (maxDistanceToFocus <= 0f)
        {
            return desiredPosition;
        }

        float maxDistance = Mathf.Max(0.1f, maxDistanceToFocus);
        Vector3 planarOffset = new Vector3(desiredPosition.x - focusPoint.x, 0f, desiredPosition.z - focusPoint.z);
        float currentDistance = planarOffset.magnitude;

        if (currentDistance <= maxDistance)
        {
            return desiredPosition;
        }

        if (currentDistance < 0.0001f)
        {
            planarOffset = Vector3.forward * maxDistance;
        }
        else
        {
            planarOffset = planarOffset / currentDistance * maxDistance;
        }

        return new Vector3(focusPoint.x + planarOffset.x, desiredPosition.y, focusPoint.z + planarOffset.z);
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

    private float ComputeHeightOffset(float horizontalDistance)
    {
        if (lookDownAngleDegrees <= 0f)
        {
            return 0f;
        }

        return Mathf.Tan(Mathf.Deg2Rad * lookDownAngleDegrees) * horizontalDistance;
    }

    private IEnumerator CaptureClipOnce()
    {
        if (targetCamera == null)
        {
            UnityEngine.Debug.LogWarning("[TunaBoidCameraRecorder] Missing camera reference.");
            yield break;
        }

        EnsureVideoRecorderReference();

        if (videoRecorder == null)
        {
            UnityEngine.Debug.LogWarning("[TunaBoidCameraRecorder] VideoRecorder not found. Recording skipped.");
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
