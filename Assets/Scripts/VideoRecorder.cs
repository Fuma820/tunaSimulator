using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System;
using UnityEngine;
using UnityEngine.Networking;

public class VideoRecorder : MonoBehaviour
{
    [SerializeField] private int recordingFrameRate = 15;               // 出力動画のフレームレート
    [SerializeField] private float recordingDuration = 5f;              // 録画時間
    private int recordingInterval = 1;                                  // 録画間隔
    private string outputDirectory = "Recordings";                      // 出力ディレクトリ
    [SerializeField] private bool stepSynchronousRecording = true;

    private bool isRecording = false;
    private int currentEpisode = 0;
    private static int globalEpisodeCounter = 0;
    private int lastRecordedEpisode = 0;                                // 最後に録画したエピソード番号
    private int activeRecordingEpisodeNumber = -1;                      // 現在録画中のエピソード番号
    private Coroutine recordingCoroutine;
    private string activeEpisodeFolder;
    private int activeFrameIndex;
    private bool sessionHasRecording;
    private RenderTexture captureRenderTexture;
    private Texture2D readbackTexture;
    private bool captureFramerateLocked;
    private int previousCaptureFramerate;

    public Camera recordingCamera;
    public int recordingWidth = 1920;
    public int recordingHeight = 1080;

    [SerializeField] private bool autoEncodeToMp4 = true;               // 保存後に自動で動画化
    [SerializeField] private string ffmpegExePath = "ffmpeg";           // ffmpeg実行ファイル
    [SerializeField] private bool deleteFramesAfterEncode = false;      // 変換成功後にPNGを削除
    [SerializeField] private bool overwriteExistingRecordings = true;
    [SerializeField] private bool lockCaptureFramerate = true;

    [SerializeField] private bool archiveUploadedVideos = true;
    [SerializeField] private string uploadArchiveSubdirectory = "Uploaded";
    [SerializeField] private bool consolidateVideosIntoSingleFolder = true;
    [SerializeField] private string consolidatedVideoSubfolder = "Videos";

    [SerializeField] private string serverBaseUrl = "http://localhost:8000";
    [SerializeField] private float serverTimeout = 180f;
    [SerializeField] private int maxRetryAttempts = 5;
    [SerializeField] private float retryDelay = 2f;
    [SerializeField] private bool enableDetailedLogging = false;
    [SerializeField] private bool uploadToServer = true;

    public System.Action<float> OnRewardReceived;
    public System.Action<int, float> OnServerScoreReceived;
    public static int GlobalEpisodeCounter => globalEpisodeCounter;
    public static System.Action<int> OnGlobalEpisodeIncremented;

    private bool lastUploadSuccess = false;
    private int currentGenerationIndex = -1;
    private int currentIndividualIndex = -1;
    private int currentSampleIndex = -1;
    private string resolvedFfmpegExecutablePath;

    private void LogStatus(string message)
    {
        Debug.Log($"[VideoRecorder] {message}");
    }

    private void BroadcastReward(int episodeNumber, float reward)
    {
        try
        {
            OnRewardReceived?.Invoke(reward);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[VideoRecorder] Reward listener threw an exception: {ex.Message}");
        }

        try
        {
            OnServerScoreReceived?.Invoke(episodeNumber, reward);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[VideoRecorder] Server score listener threw an exception: {ex.Message}");
        }
    }

    void Start()
    {
        // 録画用カメラが指定されていない場合はメインカメラを使用
        if (recordingCamera == null)
        {
            recordingCamera = Camera.main;
        }

        // 出力ディレクトリを作成
        string fullPath = Path.Combine(Application.dataPath, "..", outputDirectory);
        if (!Directory.Exists(fullPath))
        {
            Directory.CreateDirectory(fullPath);
        }
    }

    // エンコード完了通知（mp4Path, episodeNumber, success）
    public event Action<string, int, bool> OnMp4Encoded;
    // 録画セグメント完了通知（episodeNumber, capturedFrames）
    public event Action<int, int> OnRecordingSegmentCompleted;

    public bool IsRecording => isRecording;
    public bool IsBusy => recordingCoroutine != null;
    public int FramesPerSegment => (recordingFrameRate > 0 && recordingDuration > 0f)
        ? Mathf.RoundToInt(recordingFrameRate * recordingDuration)
        : 0;
    public int ActiveEpisodeNumber => activeRecordingEpisodeNumber;
    public float RecordingDurationSeconds => recordingDuration;
    public int RecordingFrameRate => recordingFrameRate;


    public void ConfigureRecordingWindow(float durationSeconds, int frameRate)
    {
        bool changed = false;

        if (durationSeconds > 0f && !Mathf.Approximately(durationSeconds, recordingDuration))
        {
            recordingDuration = durationSeconds;
            changed = true;
        }

        if (frameRate > 0 && frameRate != recordingFrameRate)
        {
            recordingFrameRate = frameRate;
            changed = true;
        }

        if (changed)
        {
            LogStatus($"Recording window enforced: {recordingDuration:F2}s @ {recordingFrameRate} FPS ({FramesPerSegment} frames).");
        }
    }

    public void SetEvaluationContext(int generationIndex, int individualIndex, int sampleIndex = -1)
    {
        currentGenerationIndex = generationIndex;
        currentIndividualIndex = individualIndex;
        currentSampleIndex = sampleIndex;
    }

    private string BuildEpisodeBaseName(int episodeNumber)
    {
        if (currentGenerationIndex >= 0 && currentIndividualIndex >= 0)
        {
            string baseName = $"Generation_{currentGenerationIndex + 1:D3}_{currentIndividualIndex + 1:D3}";
            if (currentSampleIndex >= 1)
            {
                baseName += $"_Sample_{currentSampleIndex:D2}";
            }
            return baseName;
        }

        return $"Episode_{episodeNumber:D6}";
    }

    private string BuildEpisodeFileName(int episodeNumber)
    {
        return BuildEpisodeBaseName(episodeNumber) + ".mp4";
    }

    public void NotifyEpisodeBegin()
    {
        if (currentEpisode == 0 || currentEpisode == globalEpisodeCounter)
        {
            globalEpisodeCounter++;
            currentEpisode = globalEpisodeCounter;
        
            OnGlobalEpisodeIncremented?.Invoke(globalEpisodeCounter);
        }
        
        if (!isRecording && recordingInterval > 0)
        {
            if (globalEpisodeCounter - lastRecordedEpisode >= recordingInterval)
            {
                LogStatus($"Episode {globalEpisodeCounter} reached interval. Scheduling recording.");
                StartRecordingForEpisode(globalEpisodeCounter);
                lastRecordedEpisode = globalEpisodeCounter;
            }
        }
    }

    public void StartRecording()
    {
        if (isRecording || recordingCoroutine != null)
        {
            return;
        }

        globalEpisodeCounter++;
        currentEpisode = globalEpisodeCounter;
        OnGlobalEpisodeIncremented?.Invoke(globalEpisodeCounter);

        StartRecordingForEpisode(globalEpisodeCounter);
        lastRecordedEpisode = globalEpisodeCounter;
    }

    private void StartRecordingForEpisode(int episodeNumber)
    {
        if (recordingCoroutine != null)
        {
            LogStatus($"Recorder busy. Episode {episodeNumber} will start when idle.");
            StartCoroutine(StartWhenIdleAndRecord(episodeNumber));
            return;
        }

        isRecording = true;
        activeRecordingEpisodeNumber = episodeNumber;
        PrepareEpisodeFolder(episodeNumber);
        ApplyCaptureFramerate();
        LogStatus($"Recording episode {episodeNumber} (target {FramesPerSegment} frames).");

        recordingCoroutine = StartCoroutine(RecordingCoroutine(episodeNumber));
    }

    private IEnumerator StartWhenIdleAndRecord(int episodeNumber)
    {
        while (recordingCoroutine != null)
        {
            yield return null;
        }
        StartRecordingForEpisode(episodeNumber);
    }

    private void ApplyCaptureFramerate()
    {
        if (!lockCaptureFramerate || recordingFrameRate <= 0)
        {
            return;
        }

        if (captureFramerateLocked)
        {
            return;
        }

        previousCaptureFramerate = Time.captureFramerate;
        Time.captureFramerate = recordingFrameRate;
        captureFramerateLocked = true;
    }

    private void RestoreCaptureFramerate()
    {
        if (!captureFramerateLocked)
        {
            return;
        }

        Time.captureFramerate = previousCaptureFramerate;
        captureFramerateLocked = false;
    }

    private IEnumerator RecordingCoroutine(int episodeNumber)
    {
            if (string.IsNullOrEmpty(activeEpisodeFolder))
            {
                PrepareEpisodeFolder(episodeNumber);
            }

            int captured = 0;
            int stepSyncMaxFrames = 0;
            if (recordingFrameRate > 0 && recordingDuration > 0f)
            {
                stepSyncMaxFrames = Mathf.RoundToInt(recordingFrameRate * recordingDuration);
            }

            bool usingCaptureLock = lockCaptureFramerate && recordingFrameRate > 0;
            double startRealtime = Time.realtimeSinceStartupAsDouble;
            double endRealtime = recordingDuration > 0f ? startRealtime + recordingDuration : double.PositiveInfinity;
            double frameInterval = recordingFrameRate > 0 ? 1.0 / recordingFrameRate : 0.0;
            double nextFrameTime = startRealtime + frameInterval;

            while (isRecording)
            {
                yield return new WaitForEndOfFrame();
                if (!isRecording)
                {
                    break;
                }

                CaptureFrame();
                captured++;

                bool frameLimitReached = stepSyncMaxFrames > 0 && captured >= stepSyncMaxFrames;

                if (usingCaptureLock)
                {
                    if (frameLimitReached)
                    {
                        isRecording = false;
                        break;
                    }

                    continue;
                }

                double now = Time.realtimeSinceStartupAsDouble;
                bool durationReached = now >= endRealtime;

                if (durationReached || frameLimitReached)
                {
                    if (!durationReached && recordingDuration > 0f)
                    {
                        while (isRecording && now < endRealtime)
                        {
                            double waitRemaining = endRealtime - now;
                            float waitStep = (float)Math.Min(waitRemaining, 0.1);
                            yield return new WaitForSecondsRealtime(waitStep);
                            now = Time.realtimeSinceStartupAsDouble;
                        }
                    }

                    isRecording = false;
                    break;
                }

                if (frameInterval > 0.0)
                {
                    double waitTime = nextFrameTime - now;
                    while (isRecording && waitTime > 0.0)
                    {
                        float waitStep = (float)Math.Min(waitTime, 0.1);
                        yield return new WaitForSecondsRealtime(waitStep);
                        now = Time.realtimeSinceStartupAsDouble;
                        waitTime = nextFrameTime - now;
                    }
                    nextFrameTime = Math.Max(nextFrameTime + frameInterval, now + frameInterval);
                }
                else
                {
                    yield return null;
                }
            }

            int framesRecorded = Mathf.Max(captured, activeFrameIndex);
            LogStatus($"Episode {episodeNumber} captured {framesRecorded} frames. Finalizing...");
            try
            {
                OnRecordingSegmentCompleted?.Invoke(episodeNumber, framesRecorded);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[VideoRecorder] Segment completion callback failed: {ex.Message}");
            }

            yield return StartCoroutine(FinalizeEpisodeRecording(episodeNumber));

            activeRecordingEpisodeNumber = -1;
            isRecording = false;
            recordingCoroutine = null;
            activeEpisodeFolder = null;
                RestoreCaptureFramerate();
            yield break;
    }

    private void PrepareEpisodeFolder(int episodeNumber)
    {
        string basePath = Path.Combine(Application.dataPath, "..", outputDirectory);
        if (!Directory.Exists(basePath))
        {
            Directory.CreateDirectory(basePath);
        }

        string baseFolderName = $"Episode_{episodeNumber:D6}";
        string primaryFolder = Path.Combine(basePath, baseFolderName);
        bool allowOverwriteThisRun = overwriteExistingRecordings && !sessionHasRecording;

        if (allowOverwriteThisRun)
        {
            activeEpisodeFolder = primaryFolder;
            Directory.CreateDirectory(activeEpisodeFolder);

            var existingFrames = Directory.GetFiles(activeEpisodeFolder, "frame_*.png");
            foreach (string path in existingFrames)
            {
                try
                {
                    File.Delete(path);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[VideoRecorder] Failed to delete old frame {path}: {ex.Message}");
                }
            }
        }
        else
        {
            string candidateFolder = primaryFolder;
            int suffix = 1;
            while (Directory.Exists(candidateFolder))
            {
                candidateFolder = Path.Combine(basePath, $"{baseFolderName}_{suffix:D2}");
                suffix++;
            }

            Directory.CreateDirectory(candidateFolder);
            activeEpisodeFolder = candidateFolder;
        }

        activeFrameIndex = 0;
        sessionHasRecording = true;
    }

    private string GetResolvedFfmpegPath()
    {
        if (!string.IsNullOrEmpty(resolvedFfmpegExecutablePath) && File.Exists(resolvedFfmpegExecutablePath))
        {
            return resolvedFfmpegExecutablePath;
        }

        if (!string.IsNullOrWhiteSpace(ffmpegExePath))
        {
            string trimmed = ffmpegExePath.Trim();
            if (Path.IsPathRooted(trimmed))
            {
                string expanded = ExpandHomeDirectory(trimmed);
                if (File.Exists(expanded))
                {
                    resolvedFfmpegExecutablePath = expanded;
                    return resolvedFfmpegExecutablePath;
                }
            }
            else
            {
                string fromPath = FindExecutableOnSystemPath(trimmed);
                if (!string.IsNullOrEmpty(fromPath))
                {
                    resolvedFfmpegExecutablePath = fromPath;
                    return resolvedFfmpegExecutablePath;
                }
            }
        }

        foreach (string candidate in GetDefaultFfmpegSearchPaths())
        {
            if (File.Exists(candidate))
            {
                resolvedFfmpegExecutablePath = candidate;
                return resolvedFfmpegExecutablePath;
            }
        }

        return null;
    }

    private static string ExpandHomeDirectory(string path)
    {
        if (string.IsNullOrEmpty(path) || path[0] != '~')
        {
            return path;
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
        if (string.IsNullOrEmpty(home))
        {
            return path;
        }

        return Path.Combine(home, path.Substring(1));
    }

    private static string FindExecutableOnSystemPath(string executableName)
    {
        if (string.IsNullOrWhiteSpace(executableName))
        {
            return null;
        }

        string envPath = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(envPath))
        {
            return null;
        }

        string[] paths = envPath.Split(Path.PathSeparator);
        bool needsExeExtension = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;

        foreach (string rawDir in paths)
        {
            string dir = rawDir?.Trim();
            if (string.IsNullOrEmpty(dir))
            {
                continue;
            }

            string candidate = Path.Combine(dir, executableName);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            if (needsExeExtension && !candidate.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
            {
                string exeCandidate = candidate + ".exe";
                if (File.Exists(exeCandidate))
                {
                    return exeCandidate;
                }
            }
        }

        return null;
    }

    private static IEnumerable<string> GetDefaultFfmpegSearchPaths()
    {
        yield return "/opt/homebrew/bin/ffmpeg";       // Apple Silicon Homebrew
        yield return "/usr/local/bin/ffmpeg";          // Intel Homebrew / macOS
        yield return "/usr/bin/ffmpeg";                // Linux standard package
        string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        if (!string.IsNullOrEmpty(programFiles))
        {
            yield return Path.Combine(programFiles, "ffmpeg/bin/ffmpeg.exe");
        }

        string programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        if (!string.IsNullOrEmpty(programFilesX86))
        {
            yield return Path.Combine(programFilesX86, "ffmpeg/bin/ffmpeg.exe");
        }
        yield return "C:/ffmpeg/bin/ffmpeg.exe";
    }

    private IEnumerator FinalizeEpisodeRecording(int episodeNumber)
    {
        if (string.IsNullOrEmpty(activeEpisodeFolder))
        {
            yield break;
        }

        string normalizedFolder = activeEpisodeFolder.Replace("\\", "/");
        string outputFileName = BuildEpisodeFileName(episodeNumber);
        string executableForScript = GetResolvedFfmpegPath() ?? ffmpegExePath ?? "ffmpeg";
        string ffmpegCommand = $"\"{executableForScript}\" -y -framerate {recordingFrameRate} -i \"{normalizedFolder}/frame_%06d.png\" -c:v libx264 -pix_fmt yuv420p \"{normalizedFolder}/{outputFileName}\"";
        bool isWindows = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
        string scriptName = isWindows ? "convert_to_video.bat" : "convert_to_video.sh";
        string commandFile = Path.Combine(activeEpisodeFolder, scriptName);
        string scriptBody = isWindows
            ? ffmpegCommand + "\r\npause\r\n"
            : "#!/bin/bash\n" + ffmpegCommand + "\nread -p \"Press Enter to exit\"\n";

        yield return StartCoroutine(WriteFileWithRetry(commandFile, System.Text.Encoding.UTF8.GetBytes(scriptBody), 3));

        if (!isWindows)
        {
            TryMakeScriptExecutable(commandFile);
        }

        if (autoEncodeToMp4)
        {
            LogStatus($"Encoding episode {episodeNumber} to MP4...");
            yield return StartCoroutine(EncodeToMp4Coroutine(activeEpisodeFolder, episodeNumber));
        }
    }
   
    private void CaptureFrame()
    {
        if (recordingCamera == null)
        {
            Debug.LogError("[VideoRecorder] Recording camera is not assigned. Stopping recording.");
            isRecording = false;
            return;
        }

        EnsureCaptureResources();

        RenderTexture currentRT = RenderTexture.active;
        recordingCamera.targetTexture = captureRenderTexture;
        recordingCamera.Render();

        RenderTexture.active = captureRenderTexture;
        readbackTexture.ReadPixels(new Rect(0, 0, recordingWidth, recordingHeight), 0, 0);
        readbackTexture.Apply(false);

        SaveFrameTexture(readbackTexture);

        recordingCamera.targetTexture = null;
        RenderTexture.active = currentRT;
    }

    private void SaveFrameTexture(Texture2D frame)
    {
        if (frame == null)
        {
            return;
        }

        if (string.IsNullOrEmpty(activeEpisodeFolder))
        {
            return;
        }

        byte[] bytes = frame.EncodeToPNG();
        string filename = Path.Combine(activeEpisodeFolder, $"frame_{activeFrameIndex:D6}.png");
        activeFrameIndex++;

        try
        {
            File.WriteAllBytes(filename, bytes);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[VideoRecorder] Failed to write frame {filename}: {ex.Message}");
        }
    }

    private void EnsureCaptureResources()
    {
        if (captureRenderTexture != null && (captureRenderTexture.width != recordingWidth || captureRenderTexture.height != recordingHeight))
        {
            captureRenderTexture.Release();
            Destroy(captureRenderTexture);
            captureRenderTexture = null;
        }

        if (captureRenderTexture == null)
        {
            captureRenderTexture = new RenderTexture(recordingWidth, recordingHeight, 24, RenderTextureFormat.ARGB32)
            {
                name = "VideoRecorder_CaptureRT"
            };
            captureRenderTexture.Create();
        }

        if (readbackTexture != null && (readbackTexture.width != recordingWidth || readbackTexture.height != recordingHeight))
        {
            Destroy(readbackTexture);
            readbackTexture = null;
        }

        if (readbackTexture == null)
        {
            readbackTexture = new Texture2D(recordingWidth, recordingHeight, TextureFormat.RGB24, false, false)
            {
                name = "VideoRecorder_Readback"
            };
        }
    }

    private void CleanupCaptureResources()
    {
        if (captureRenderTexture != null)
        {
            captureRenderTexture.Release();
            Destroy(captureRenderTexture);
            captureRenderTexture = null;
        }

        if (readbackTexture != null)
        {
            Destroy(readbackTexture);
            readbackTexture = null;
        }
    }

        private void TryMakeScriptExecutable(string scriptPath)
        {
            if (string.IsNullOrEmpty(scriptPath))
            {
                return;
            }

            try
            {
                var chmodInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "/bin/chmod",
                    Arguments = $"+x \"{scriptPath}\"",
                    CreateNoWindow = true,
                    UseShellExecute = false,
                    RedirectStandardError = false,
                    RedirectStandardOutput = false
                };

                var process = System.Diagnostics.Process.Start(chmodInfo);
                if (process != null)
                {
                    process.WaitForExit(200);
                    process.Dispose();
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[VideoRecorder] Failed to set execute permission on script {scriptPath}: {ex.Message}");
            }
        }

    private IEnumerator WriteFileWithRetry(string filePath, byte[] data, int maxRetries)
    {
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            bool success = false;
            string errorMessage = "";
            
            if (File.Exists(filePath))
            {
                yield return StartCoroutine(WaitForFileAccess(filePath, 2.0f));
            }
            
            try
            {
                File.WriteAllBytes(filePath, data);
                success = true;
            }
            catch (System.IO.IOException ex)
            {
                errorMessage = ex.Message;
            }
            catch (System.Exception ex)
            {
                errorMessage = ex.Message;
                Debug.LogError($"[VideoRecorder] Unexpected error writing file: {errorMessage}");
                yield break;
            }
            
            if (success)
            {
                yield break;
            }
            
            if (attempt < maxRetries - 1)
            {
                yield return new WaitForSeconds(0.5f + attempt * 0.5f);
            }
            else
            {
                Debug.LogError($"[VideoRecorder] Failed to write file after {maxRetries} attempts: {filePath}");
            }
        }
    }

    private IEnumerator WaitForFileAccess(string filePath, float timeoutSeconds)
    {
        float startTime = Time.realtimeSinceStartup;
        
        while (Time.realtimeSinceStartup - startTime < timeoutSeconds)
        {
            bool canAccess = false;
            
            try
            {
                using (FileStream fs = File.Open(filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None))
                {
                    canAccess = true;
                }
            }
            catch (System.IO.IOException)
            {
                canAccess = false;
            }
            
            if (canAccess)
            {
                yield break;
            }
            
            yield return new WaitForSeconds(0.1f);
        }
        
        Debug.LogWarning($"[VideoRecorder] File access timeout: {filePath}");
    }

    /// <summary>
    /// FFmpegでPNGからMP4を作成
    /// </summary>
    private IEnumerator EncodeToMp4Coroutine(string episodeFolder, int episodeNumber)
    {
        string pattern = Path.Combine(episodeFolder, "frame_%06d.png");
        string outputMp4 = Path.Combine(episodeFolder, BuildEpisodeFileName(episodeNumber));

        string executablePath = GetResolvedFfmpegPath();
        if (string.IsNullOrEmpty(executablePath))
        {
            Debug.LogError("[VideoRecorder] FFmpeg executable not found. Set 'ffmpegExePath' or install ffmpeg so recordings can be converted automatically.");
            yield break;
        }

        if (File.Exists(outputMp4))
        {
            yield return StartCoroutine(DeleteFileWithRetry(outputMp4, 5));
            yield return StartCoroutine(WaitForFileAccess(outputMp4, 3.0f));
        }

        yield return new WaitForSeconds(0.2f);

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = executablePath,
            Arguments = $"-y -f image2 -framerate {recordingFrameRate} -i \"{pattern}\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \"{outputMp4}\"",
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = episodeFolder
        };

        System.Diagnostics.Process proc;
        string startError;
        if (!TryStartExternalProcess(psi, out proc, out startError))
        {
            Debug.LogError($"Failed to start ffmpeg: {startError}. Path: {ffmpegExePath}");
            yield break;
        }

        try
        {
            float startTime = Time.realtimeSinceStartup;
            float timeout = 60f;
            
            while (!proc.HasExited)
            {
                if (Time.realtimeSinceStartup - startTime > timeout)
                {
                    Debug.LogError($"FFmpeg timeout after {timeout} seconds. Killing process.");
                    try
                    {
                        proc.Kill();
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError($"Failed to kill FFmpeg process: {ex.Message}");
                    }
                    yield break;
                }
                yield return null;
            }

            if (proc.ExitCode == 0)
            {
                LogStatus($"Episode {episodeNumber} encoded successfully.");
                
                yield return new WaitForSeconds(0.5f);
                
                if (deleteFramesAfterEncode)
                {
                    yield return StartCoroutine(CleanupFrameFiles(episodeFolder));
                }
                
                string finalOutputPath = outputMp4;
                if (consolidateVideosIntoSingleFolder)
                {
                    string relocatedPath = TryRelocateVideoToConsolidatedFolder(outputMp4);
                    if (!string.IsNullOrEmpty(relocatedPath))
                    {
                        finalOutputPath = relocatedPath;
                    }
                }

                // エンコード成功を通知
                OnMp4Encoded?.Invoke(finalOutputPath, episodeNumber, true);
                
                if (uploadToServer)
                {
                    StartCoroutine(UploadVideoToServer(finalOutputPath, episodeNumber));
                }
            }
            else
            {
                string errorOutput = "";
                try
                {
                    errorOutput = proc.StandardError.ReadToEnd();
                }
                catch (System.Exception ex)
                {
                    Debug.LogWarning($"Failed to read FFmpeg error output: {ex.Message}");
                }

                Debug.LogError($"FFmpeg failed with exit code {proc.ExitCode}. Error output: {errorOutput}");
                LogStatus($"Episode {episodeNumber} encoding failed (exit {proc.ExitCode}).");
                
                if (errorOutput.Contains("device file") || errorOutput.Contains("access") || errorOutput.Contains("使用中"))
                {
                    Debug.LogWarning("File access conflict detected. Retrying may resolve the issue.");
                }
                
                // エンコード失敗を通知
                OnMp4Encoded?.Invoke(outputMp4, episodeNumber, false);
            }
        }
        finally
        {
            if (proc != null) 
            {
                try
                {
                    if (!proc.HasExited)
                    {
                        proc.Kill();
                    }
                    proc.Dispose();
                }
                catch (System.Exception ex)
                {
                    Debug.LogWarning($"Error disposing FFmpeg process: {ex.Message}");
                }
            }
        }
    }

    private IEnumerator DeleteFileWithRetry(string filePath, int maxRetries)
    {
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            bool success = false;
            string errorMessage = "";
            
            try
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
                success = true;
            }
            catch (System.IO.IOException ex)
            {
                errorMessage = ex.Message;
            }
            
            if (success)
            {
                yield break;
            }
            
            if (attempt < maxRetries - 1)
            {
                yield return new WaitForSeconds(0.5f + attempt * 0.5f);
            }
            else
            {
                Debug.LogError($"[VideoRecorder] Failed to delete file after {maxRetries} attempts: {filePath}");
            }
        }
    }

    private IEnumerator CleanupFrameFiles(string episodeFolder)
    {
        var frameFiles = Directory.GetFiles(episodeFolder, "frame_*.png");
        int deletedCount = 0;
        
        foreach (var filePath in frameFiles)
        {
            bool deleted = false;
            
            try
            {
                File.Delete(filePath);
                deleted = true;
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[VideoRecorder] Failed to delete frame file {filePath}: {ex.Message}");
            }
            
            if (deleted)
            {
                deletedCount++;
                
                // 10ファイルごとに1フレーム待機
                if (deletedCount % 10 == 0)
                {
                    yield return null;
                }
            }
        }
    }

    private string TryRelocateVideoToConsolidatedFolder(string sourcePath)
    {
        if (string.IsNullOrEmpty(sourcePath))
        {
            return sourcePath;
        }

        if (!File.Exists(sourcePath))
        {
            Debug.LogWarning($"[VideoRecorder] Consolidation skipped. Source missing: {sourcePath}");
            return sourcePath;
        }

        string recordingsRoot = Path.Combine(Application.dataPath, "..", outputDirectory);
        string targetDirectory = recordingsRoot;
        if (!string.IsNullOrWhiteSpace(consolidatedVideoSubfolder))
        {
            targetDirectory = Path.Combine(recordingsRoot, consolidatedVideoSubfolder.Trim());
        }

        try
        {
            Directory.CreateDirectory(targetDirectory);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[VideoRecorder] Failed to create consolidated folder '{targetDirectory}': {ex.Message}");
            return sourcePath;
        }

        string destinationPath = Path.Combine(targetDirectory, Path.GetFileName(sourcePath));

        try
        {
            if (File.Exists(destinationPath))
            {
                File.Delete(destinationPath);
            }

            File.Move(sourcePath, destinationPath);
            return destinationPath;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[VideoRecorder] Failed to relocate MP4 to '{destinationPath}': {ex.Message}");
            return sourcePath;
        }
    }

    private bool TryStartExternalProcess(System.Diagnostics.ProcessStartInfo psi, out System.Diagnostics.Process proc, out string error)
    {
        proc = null;
        error = null;
        try
        {
            proc = new System.Diagnostics.Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.Log($"ffmpeg: {e.Data}"); };
            proc.ErrorDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.Log($"ffmpeg: {e.Data}"); };
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            return true;
        }
        catch (System.Exception ex)
        {
            error = ex.Message;
            try { if (proc != null) proc.Dispose(); } catch { }
            proc = null;
            return false;
        }
    }

    public void StopRecording()
    {
        if (!isRecording) return;
        isRecording = false;
    }

    void OnDestroy()
    {
        if (isRecording)
        {
            isRecording = false;
        }

        if (recordingCoroutine != null)
        {
            StopCoroutine(recordingCoroutine);
            recordingCoroutine = null;
        }

        System.GC.Collect();

        CleanupCaptureResources();
        RestoreCaptureFramerate();
        sessionHasRecording = false;
    }

    #region Server Communication

    private IEnumerator UploadVideoToServer(string videoPath, int episodeNumber)
    {
        if (enableDetailedLogging)
        {
            Debug.Log($"[VideoRecorder] Starting upload for episode {episodeNumber}: {videoPath}");
        }
        else
        {
            LogStatus($"Uploading episode {episodeNumber} to server...");
        }

        if (!File.Exists(videoPath))
        {
            Debug.LogError($"[VideoRecorder] Video file not found: {videoPath}");
            yield break;
        }

        PersistUploadedVideo(videoPath, episodeNumber);

        byte[] videoData;
        try
        {
            videoData = File.ReadAllBytes(videoPath);
            if (enableDetailedLogging)
            {
                Debug.Log($"[VideoRecorder] Video file size: {videoData.Length / 1024.0f / 1024.0f:F2} MB");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[VideoRecorder] Failed to read video file: {ex.Message}");
            yield break;
        }

        for (int attempt = 1; attempt <= maxRetryAttempts; attempt++)
        {
            if (enableDetailedLogging)
            {
                Debug.Log($"[VideoRecorder] Upload attempt {attempt}/{maxRetryAttempts}");
            }
            else
            {
                LogStatus($"Upload attempt {attempt}/{maxRetryAttempts} for episode {episodeNumber}.");
            }

            string uploadFileName = Path.GetFileName(videoPath);
            if (string.IsNullOrEmpty(uploadFileName))
            {
                uploadFileName = BuildEpisodeFileName(episodeNumber);
            }

            yield return StartCoroutine(TryUploadVideo(videoData, episodeNumber, attempt, uploadFileName));

            if (lastUploadSuccess)
            {
                break;
            }

            if (attempt < maxRetryAttempts)
            {
                if (enableDetailedLogging)
                {
                    Debug.Log($"[VideoRecorder] Retrying in {retryDelay} seconds...");
                }
                yield return new WaitForSeconds(retryDelay);
            }
        }

        if (!lastUploadSuccess)
        {
            Debug.LogError($"[VideoRecorder] Failed to upload video after {maxRetryAttempts} attempts");
        }
    }

    private IEnumerator TryUploadVideo(byte[] videoData, int episodeNumber, int attemptNumber, string uploadFileName)
    {
        lastUploadSuccess = false;

        string url = $"{serverBaseUrl}/upload/video";

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", videoData, uploadFileName, "video/mp4");
        form.AddField("episode_number", episodeNumber.ToString());
        form.AddField("attempt_number", attemptNumber.ToString());
        form.AddField("file_size", videoData.Length.ToString());

        using (UnityWebRequest request = UnityWebRequest.Post(url, form))
        {
            request.timeout = (int)serverTimeout;

            if (enableDetailedLogging)
            {
                Debug.Log($"[VideoRecorder] Sending request to: {url} (Episode: {episodeNumber}, Size: {videoData.Length / 1024.0f / 1024.0f:F2} MB)");
            }

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    string responseText = request.downloadHandler.text;

                    var response = JsonUtility.FromJson<ServerResponse>(responseText);
                    
                    if (response != null && response.status == "ok")
                    {
                        lastUploadSuccess = true;

                        bool responseHasScoreField = responseText.Contains("\"score\"");
                        float resolvedReward = responseHasScoreField ? response.score : response.reward;

                        BroadcastReward(episodeNumber, resolvedReward);

                        if (OnRewardReceived == null && OnServerScoreReceived == null)
                        {
                            Debug.LogWarning($"[VideoRecorder] No listeners subscribed to receive reward for episode {episodeNumber}");
                        }
                        else if (enableDetailedLogging)
                        {
                            Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Reward {resolvedReward} distributed to listeners");
                        }
                        else
                        {
                            LogStatus($"Episode {episodeNumber} upload succeeded. Reward {resolvedReward:F3}.");
                        }
                    }
                    else
                    {
                        Debug.LogError($"[VideoRecorder] Server returned error status for episode {episodeNumber}: {responseText}");
                        
                        BroadcastReward(episodeNumber, 0f);
                        Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Distributed default reward 0 due to server error");
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[VideoRecorder] Failed to parse server response for episode {episodeNumber}: {ex.Message}");
                    
                    BroadcastReward(episodeNumber, 0f);
                    Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Distributed default reward 0 due to parse error");
                }
            }
            else
            {
                Debug.LogError($"[VideoRecorder] Upload failed (attempt {attemptNumber}): {request.error}");
                LogStatus($"Episode {episodeNumber} upload attempt {attemptNumber} failed.");
                if (enableDetailedLogging)
                {
                    Debug.Log($"[VideoRecorder] Response code: {request.responseCode}");
                    if (request.downloadHandler != null && !string.IsNullOrEmpty(request.downloadHandler.text))
                    {
                        Debug.Log($"[VideoRecorder] Response body: {request.downloadHandler.text}");
                    }
                }

                BroadcastReward(episodeNumber, 0f);
                Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Distributed default reward 0 due to network error");
            }
        }
    }

    [System.Serializable]
    private class ServerResponse
    {
        public string status;
        public int episode_number;
        public float reward;
        public float score;          // 平均コサイン類似度
    }

    private void PersistUploadedVideo(string sourcePath, int episodeNumber)
    {
        if (!archiveUploadedVideos)
        {
            return;
        }

        try
        {
            string baseDirectory = Path.Combine(Application.dataPath, "..", outputDirectory ?? "Recordings");
            Directory.CreateDirectory(baseDirectory);

            string subDirectory = string.IsNullOrWhiteSpace(uploadArchiveSubdirectory)
                ? baseDirectory
                : Path.Combine(baseDirectory, uploadArchiveSubdirectory.Trim());
            Directory.CreateDirectory(subDirectory);

            string originalName = Path.GetFileNameWithoutExtension(sourcePath);
            if (string.IsNullOrEmpty(originalName))
            {
                originalName = BuildEpisodeBaseName(episodeNumber);
            }

            string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmssfff", CultureInfo.InvariantCulture);
            string destinationName = $"{originalName}_{timestamp}.mp4";
            string destinationPath = Path.Combine(subDirectory, destinationName);

            File.Copy(sourcePath, destinationPath, overwrite: false);
            LogStatus($"Archived upload for episode {episodeNumber} -> {destinationPath}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[VideoRecorder] Failed to archive uploaded video: {ex.Message}");
        }
    }

    #endregion
    
}
