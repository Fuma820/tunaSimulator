using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System;
using UnityEngine;
using UnityEngine.Networking;

public class VideoRecorder : MonoBehaviour
{
    [Header("Recording Settings")]
    [SerializeField, Tooltip("出力動画のフレームレート（FFmpegに渡す入力フレームレート）。実行時のフレームレートとは独立して30FPSなどに固定できます。")]
    private int recordingFrameRate = 15;          // 出力動画のフレームレート（エンコード時）
    [SerializeField, Tooltip("動画の目的秒数。stepSynchronousRecording時の保存枚数は recordingFrameRate × recordingDuration で自動算出します。")]
    private float recordingDuration = 5f;        // 録画時間（秒）
    private int recordingInterval = 1;          // 録画間隔（エピソード数）
    private string outputDirectory = "Recordings"; // 出力ディレクトリ
    [SerializeField, Tooltip("ステップ同期録画（各フレーム=各計算ステップをすべて保存）。StopRecording() が呼ばれるまで毎フレーム保存します。")]
    private bool stepSynchronousRecording = true;
    // stepSyncMaxFrames は recordingFrameRate × recordingDuration で自動算出します（0 以下で無制限にしたい場合は recordingDuration <= 0 か recordingFrameRate <= 0 に設定）
    // タイムスケール関連の強制は行わない（録画は実時間ベースで進行）

    private bool isRecording = false;
    private int currentEpisode = 0;       // 実際のエピソード番号（NotifyEpisodeBeginで加算）
    private static int globalEpisodeCounter = 0;  // 全エージェント共通のグローバルエピソードカウンター
    private int lastRecordedEpisode = 0;  // 最後に録画したエピソード番号
    private int activeRecordingEpisodeNumber = -1; // 現在録画中のエピソード番号（手動停止時に使用）
    private Coroutine recordingCoroutine;
    private string activeEpisodeFolder;
    private int activeFrameIndex;
    private bool sessionHasRecording;
    private RenderTexture captureRenderTexture;
    private Texture2D readbackTexture;
    private bool captureFramerateLocked;
    private int previousCaptureFramerate;
    // タイムスケールの保存/復帰は行わない

    [Header("Camera Settings")]
    public Camera recordingCamera;               // 録画用カメラ
    public int recordingWidth = 1920;           // 録画解像度（幅）
    public int recordingHeight = 1080;          // 録画解像度（高さ）

    [Header("FFmpeg Encoding")]
    [SerializeField] private bool autoEncodeToMp4 = true;          // 保存後に自動で動画化する
    [SerializeField] private string ffmpegExePath = "ffmpeg";       // ffmpeg実行ファイル（PATHにあるなら"ffmpeg"のままで可）
    [SerializeField] private bool deleteFramesAfterEncode = false;  // 変換成功後にPNGを削除
    [SerializeField, Tooltip("既存のエピソードフォルダを上書きするかどうか")] private bool overwriteExistingRecordings = true;
    [SerializeField, Tooltip("録画中は Time.captureFramerate を recordingFrameRate に固定するか")] private bool lockCaptureFramerate = true;

    [Header("Recording Persistence")]
    [SerializeField, Tooltip("サーバーに送信する MP4 を Recordings フォルダ配下にアーカイブ保存するかどうか。")] private bool archiveUploadedVideos = true;
    [SerializeField, Tooltip("アーカイブを保存するサブフォルダ名（Recordings/以下）")] private string uploadArchiveSubdirectory = "Uploaded";
    [SerializeField, Tooltip("全てのMP4をひとつのサブフォルダに集約保存するかどうか。")] private bool consolidateVideosIntoSingleFolder = true;
    [SerializeField, Tooltip("集約保存するサブフォルダ名（Recordings/以下）。空欄で Recordings 直下へ保存。")] private string consolidatedVideoSubfolder = "Videos";

    [Header("Server Communication")]
    [SerializeField] private string serverBaseUrl = "http://localhost:8000";
    [SerializeField] private float serverTimeout = 180f; // Timeout for server requests
    [SerializeField] private int maxRetryAttempts = 5;
    [SerializeField] private float retryDelay = 2f;
    [SerializeField] private bool enableDetailedLogging = false; // デフォルトでログを無効化
    [SerializeField, Tooltip("Disable to skip uploading recordings to the configured server.")] private bool uploadToServer = true;

    // Event for distributing rewards to all agents
    public System.Action<float> OnRewardReceived;
    public System.Action<int, float> OnServerScoreReceived;

    // Public property to access global episode counter
    public static int GlobalEpisodeCounter => globalEpisodeCounter;

    // Event for notifying episode increment
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

    /// <summary>
    /// Override the recording window so external controllers can ensure consistent clip lengths.
    /// </summary>
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

    /// <summary>
    /// エピソード開始時に呼び出してください（MultiFishAgent.OnEpisodeBeginから）
    /// 設定された録画間隔(recordingInterval)ごとに録画を開始します。
    /// </summary>
    public void NotifyEpisodeBegin()
    {
        // グローバルエピソードカウンターをインクリメント（最初の呼び出しのみ）
        if (currentEpisode == 0 || currentEpisode == globalEpisodeCounter)
        {
            globalEpisodeCounter++;
            currentEpisode = globalEpisodeCounter;
            
            // グローバルエピソード更新ログは削除（通常動作のため不要）
            
            // 全エージェントにグローバルエピソード更新を通知
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

    /// <summary>
    /// 録画を開始
    /// </summary>
    public void StartRecording()
    {
        // 手動強制録画用（GUIボタンなど）。既存の録画や保存処理が完了していることを確認してから開始する。
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

    // 内部ヘルパー：指定のエピソード番号で録画を開始
    private void StartRecordingForEpisode(int episodeNumber)
    {
        // 録画開始ログは削除（通常動作のため不要）

        // タイムスケールの変更は行わない

        // 既に録画・保存処理中であれば、完了後に開始するよう待機を予約
        if (recordingCoroutine != null)
        {
            // 録画ビジーログは削除（通常動作のため不要）
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

    // 現在の録画/保存が完了するまで待ってから録画を開始
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

    /// <summary>
    /// 録画処理のコルーチン（指定時間録画）
    /// </summary>
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

                    // capture lock advances game time at the target FPS, so no extra waiting needed
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
            // 録画完了ログは削除（通常動作のため不要）
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

        // First try the user-provided path
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
        // Common install locations for ffmpeg across OSes
        yield return "/opt/homebrew/bin/ffmpeg";        // Apple Silicon Homebrew
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
    // Upload (and reward handling) is intentionally not implemented here to avoid duplication with MultiFishAgent.

    /// <summary>
    /// フレームをキャプチャしてリストに追加
    /// </summary>
    private void CaptureFrame()
    {
        if (recordingCamera == null)
        {
            Debug.LogError("[VideoRecorder] Recording camera is not assigned. Stopping recording.");
            isRecording = false;
            return;
        }

        EnsureCaptureResources();

        // RenderTextureを作成
        RenderTexture currentRT = RenderTexture.active;
        recordingCamera.targetTexture = captureRenderTexture;
        recordingCamera.Render();

        // RenderTextureからTexture2Dに変換
        RenderTexture.active = captureRenderTexture;
        readbackTexture.ReadPixels(new Rect(0, 0, recordingWidth, recordingHeight), 0, 0);
        readbackTexture.Apply(false);

        SaveFrameTexture(readbackTexture);

        // クリーンアップ
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

    /// <summary>
    /// ファイル書き込みをリトライ機能付きで実行
    /// </summary>
    private IEnumerator WriteFileWithRetry(string filePath, byte[] data, int maxRetries)
    {
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            bool success = false;
            string errorMessage = "";
            
            // ファイルアクセス待機は try-catch の外で実行
            if (File.Exists(filePath))
            {
                // 既存ファイルがロックされていないかチェック
                yield return StartCoroutine(WaitForFileAccess(filePath, 2.0f));
            }
            
            try
            {
                File.WriteAllBytes(filePath, data);
                success = true;
                
                // ファイル書き込み成功ログは削除（通常動作のため不要）
            }
            catch (System.IO.IOException ex)
            {
                errorMessage = ex.Message;
                // ファイル書き込み失敗の個別ログは削除（最終結果のみログ出力）
            }
            catch (System.Exception ex)
            {
                errorMessage = ex.Message;
                Debug.LogError($"[VideoRecorder] Unexpected error writing file: {errorMessage}");
                yield break;
            }
            
            if (success)
            {
                yield break; // 成功したら終了
            }
            
            if (attempt < maxRetries - 1)
            {
                // 次の試行前に少し待機
                yield return new WaitForSeconds(0.5f + attempt * 0.5f);
            }
            else
            {
                Debug.LogError($"[VideoRecorder] Failed to write file after {maxRetries} attempts: {filePath}");
            }
        }
    }

    /// <summary>
    /// ファイルアクセス可能になるまで待機
    /// </summary>
    private IEnumerator WaitForFileAccess(string filePath, float timeoutSeconds)
    {
        float startTime = Time.realtimeSinceStartup;
        
        while (Time.realtimeSinceStartup - startTime < timeoutSeconds)
        {
            bool canAccess = false;
            
            try
            {
                // ファイルが使用可能かテスト
                using (FileStream fs = File.Open(filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None))
                {
                    // ファイルにアクセス可能
                    canAccess = true;
                }
            }
            catch (System.IO.IOException)
            {
                // ファイルが使用中
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
    /// FFmpegでPNGからMP4を作成する
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

        // 既存のMP4ファイルがある場合は削除を試行
        if (File.Exists(outputMp4))
        {
            yield return StartCoroutine(DeleteFileWithRetry(outputMp4, 5));
            // ファイル削除後、確実にアクセス可能になるまで待機
            yield return StartCoroutine(WaitForFileAccess(outputMp4, 3.0f));
        }

        // 一時的なファイルロックがないことを確認
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

        // ここでプロセス起動時の例外を吸収（catchはコルーチン外のヘルパーで）
        System.Diagnostics.Process proc;
        string startError;
        if (!TryStartExternalProcess(psi, out proc, out startError))
        {
            Debug.LogError($"Failed to start ffmpeg: {startError}. Path: {ffmpegExePath}");
            yield break;
        }

        try
        {
            // 完了待ち（非ブロッキングに1フレームずつ待機）
            float startTime = Time.realtimeSinceStartup;
            float timeout = 60f; // 60秒タイムアウト
            
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
                // FFmpeg成功ログは削除（通常動作のため不要）
                LogStatus($"Episode {episodeNumber} encoded successfully.");
                
                // ファイルが確実に作成されるまで少し待機
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
                
                // Upload to server and distribute reward when enabled
                if (uploadToServer)
                {
                    StartCoroutine(UploadVideoToServer(finalOutputPath, episodeNumber));
                }
            }
            else
            {
                // FFmpegの標準エラー出力を取得してログに出力
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
                
                // ファイルアクセス関連のエラーかチェック
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

    /// <summary>
    /// ファイル削除をリトライ機能付きで実行
    /// </summary>
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
                    // ファイル削除成功ログは削除（通常動作のため不要）
                }
                success = true;
            }
            catch (System.IO.IOException ex)
            {
                errorMessage = ex.Message;
                // ファイル削除失敗の個別ログは削除（最終結果のみログ出力）
            }
            
            if (success)
            {
                yield break; // 成功したら終了
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

    /// <summary>
    /// フレームファイルのクリーンアップ
    /// </summary>
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
        
        // クリーンアップ完了ログは削除（通常動作のため不要）
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

    /// <summary>
    /// 例外を内部で捕捉して外部プロセスを起動（CS1626回避のため、yieldを含まないヘルパー）
    /// </summary>
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

    /// <summary>
    /// 手動で録画を停止
    /// </summary>
    public void StopRecording()
    {
        if (!isRecording) return;
        isRecording = false;
    }

    void OnDestroy()
    {
        // アクティブな録画を停止
        if (isRecording)
        {
            isRecording = false;
        }

        // アクティブなコルーチンを停止
        if (recordingCoroutine != null)
        {
            StopCoroutine(recordingCoroutine);
            recordingCoroutine = null;
        }

        // 実行中のFFmpegプロセスがあれば終了を試行
        System.GC.Collect(); // ガベージコレクションを強制実行してファイルハンドルを解放

        CleanupCaptureResources();
        RestoreCaptureFramerate();
        sessionHasRecording = false;
    }

    // Debug OnGUI removed for production simplicity.

    #region Server Communication

    /// <summary>
    /// Upload video to server and distribute reward to all agents
    /// </summary>
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

        // Check if file exists
        if (!File.Exists(videoPath))
        {
            Debug.LogError($"[VideoRecorder] Video file not found: {videoPath}");
            yield break;
        }

        PersistUploadedVideo(videoPath, episodeNumber);

        // Read video file
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

        // Attempt upload with retries
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
                break; // Success, exit retry loop
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

    /// <summary>
    /// Single upload attempt
    /// </summary>
    private IEnumerator TryUploadVideo(byte[] videoData, int episodeNumber, int attemptNumber, string uploadFileName)
    {
        lastUploadSuccess = false;

        string url = $"{serverBaseUrl}/upload/video"; // 正しいエンドポイント

        // Create form data - サーバーの期待する形式に合わせる
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
                    // サーバーレスポンスログは削除（通常動作のため不要）

                    // Parse JSON response - サーバーの実際のレスポンス形式に合わせる
                    var response = JsonUtility.FromJson<ServerResponse>(responseText);
                    
                    if (response != null && response.status == "ok")
                    {
                        lastUploadSuccess = true;

                        bool responseHasScoreField = responseText.Contains("\"score\"");
                        float resolvedReward = responseHasScoreField ? response.score : response.reward;

                        // Distribute reward (score) to all subscribed agents
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
                        
                        // エラーの場合でも同期のため0報酬を配信
                        BroadcastReward(episodeNumber, 0f);
                        Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Distributed default reward 0 due to server error");
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[VideoRecorder] Failed to parse server response for episode {episodeNumber}: {ex.Message}");
                    
                    // パース失敗でも同期のため0報酬を配信
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
                
                // ネットワークエラーの場合でも同期のため0報酬を配信
                BroadcastReward(episodeNumber, 0f);
                Debug.Log($"[VideoRecorder] Episode {episodeNumber}: Distributed default reward 0 due to network error");
            }
        }
    }

    /// <summary>
    /// Server response data structure - サーバーの実際のレスポンス形式に合わせる
    /// </summary>
    [System.Serializable]
    private class ServerResponse
    {
        public string status;        // "ok" or error status
        public int episode_number;   // サーバーから返されるエピソード番号
        public float reward;         // 計算された報酬値
        public float score;          // server_videomae_api 互換: 平均コサイン類似度
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
