using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 指定されたオブジェクトをランダムな位置に生成するためのクラス
/// </summary>
public class ObjectGenerator : MonoBehaviour
{
    public GameObject objectToSpawn;
    public int numberOfObjects = 5;     // 生成するオブジェクトの数
    public float spawnRadius = 5f;     // 現在位置からの半径
    public float spawnHeight = 10f;     // 円柱の高さ
    public int maxAttempts = 100;       // 生成位置の最大試行回数
    public bool debugMode = false;      // デバッグモード

    private readonly List<Vector3> spawnedPositions = new List<Vector3>(); // 既に生成された位置のリスト
    private bool spawnCompleted;

    public IReadOnlyList<Vector3> SpawnedPositions => spawnedPositions; // 生成済みローカル座標
    public bool HasCompletedSpawning => spawnCompleted;
    public Vector3 AverageSpawnPositionLocal
    {
        get
        {
            if (spawnedPositions.Count == 0)
            {
                return Vector3.zero;
            }

            Vector3 sum = Vector3.zero;
            for (int i = 0; i < spawnedPositions.Count; i++)
            {
                sum += spawnedPositions[i];
            }

            return sum / spawnedPositions.Count;
        }
    }

    public Vector3 AverageSpawnPositionWorld => transform.TransformPoint(AverageSpawnPositionLocal);

    void Start()
    {
        SpawnObjects(objectToSpawn, numberOfObjects);
    }

    /// <summary>
    /// オブジェクトを生成するメソッド
    ///
    /// 現在位置からspawnRadius以内のランダムな位置にcount個生成する．
    /// y座標は常に正（地面より下に生成されない)となる．
    /// 他のオブジェクトとminDistance以上離れた位置に生成する．
    /// </summary>
    /// <param name="obj">生成するプレハブ</param>
    /// <param name="num">生成数</param>
    void SpawnObjects(GameObject obj, int num)
    {
        Vector3 centerPosition = Vector3.zero; // ローカル座標の原点
        int successCount = 0;
        spawnedPositions.Clear(); // リストをクリア
        spawnCompleted = false;
        
        if (debugMode)
        {
            Debug.Log($"[ObjectGenerator] World position of generator: {transform.position}");
        }

        for (int i = 0; i < num; i++)
        {
            Vector3 spawnPosition = Vector3.zero;
            bool validPositionFound = false;

            // 有効な位置が見つかるまで試行
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                // 範囲内のランダムな位置を生成
                Vector3 candidatePosition = GenerateRandomPositionInRadius(centerPosition, spawnRadius);

                // BoxColliderとの当たり判定でチェック
                if (IsPositionValid(candidatePosition))
                {
                    spawnPosition = candidatePosition;
                    validPositionFound = true;
                    
                    if (debugMode)
                    {
                        float distanceFromCenter = Vector3.Distance(centerPosition, candidatePosition);
                        Debug.Log($"[ObjectGenerator] Object {i + 1}/{num}: Found valid position at {spawnPosition}, distance from center: {distanceFromCenter:F2}m (attempt {attempt + 1})");
                    }
                    break;
                }
            }

            // 有効な位置が見つかった場合のみ生成
            if (validPositionFound)
            {
                // ローカル座標をワールド座標に変換して生成
                Vector3 worldPosition = transform.TransformPoint(spawnPosition);
                GameObject spawnedObj = Instantiate(obj, worldPosition, Quaternion.identity, transform);
                spawnedPositions.Add(spawnPosition);
                successCount++;
                
                // 生成された座標をログに記録（ローカル座標とワールド座標の両方）
                Debug.Log($"[ObjectGenerator] Object {i + 1}/{num} spawned at local: {spawnPosition}, world: {worldPosition}");
            }
            else
            {
                Debug.LogWarning($"[ObjectGenerator] Object {i + 1}/{num}: Failed to find valid position after {maxAttempts} attempts. Skipping spawn.");
            }
        }
        
        // 最終サマリーログ - すべての生成座標を出力
        Debug.Log($"[ObjectGenerator] Spawn completed. Successfully spawned: {successCount}/{num} objects");
        Debug.Log($"[ObjectGenerator] Final spawned positions: {string.Join(", ", spawnedPositions)}");
        spawnCompleted = true;
    }

    /// <summary>
    /// 範囲内のランダムな位置を生成（円柱状）
    /// </summary>
    private Vector3 GenerateRandomPositionInRadius(Vector3 center, float radius) {
        // 円柱状の範囲内でランダムな位置を生成
        // XZ平面上で円形にランダムな位置を取得
        float angle = Random.Range(0f, 2f * Mathf.PI);
        float distance = Random.Range(0f, radius);
        
        Vector3 randomOffset = new Vector3(
            Mathf.Cos(angle) * distance,
            Random.Range(0f, spawnHeight), // Y座標は0からspawnHeightの範囲
            Mathf.Sin(angle) * distance
        );
        
        Vector3 randomPoint = center + randomOffset;
        return randomPoint;
    }

    /// <summary>
    /// 指定位置が他のColliderと衝突していないかチェック（全Colliderタイプ対応）
    /// </summary>
    private bool IsPositionValid(Vector3 position)
    {
        // 生成するオブジェクトのColliderを取得（BoxColliderを優先）
        BoxCollider boxCollider = objectToSpawn.GetComponent<BoxCollider>();
        Collider objCollider = boxCollider != null ? boxCollider : objectToSpawn.GetComponent<Collider>();
        
        if (objCollider == null)
        {
            Debug.LogWarning("[ObjectGenerator] Object prefab does not have a Collider. Skipping collision check.");
            return true;
        }

        Vector3 worldPosition = transform.TransformPoint(position);
        Collider[] hitColliders = null;

        // Colliderの種類に応じた衝突判定
        if (objCollider is BoxCollider boxCol)
        {
            Vector3 boxSize = Vector3.Scale(boxCol.size, objectToSpawn.transform.localScale);
            Vector3 boxCenter = position + boxCol.center;
            Vector3 worldBoxCenter = transform.TransformPoint(boxCenter);
            hitColliders = Physics.OverlapBox(worldBoxCenter, boxSize / 2f, Quaternion.identity, -1, QueryTriggerInteraction.Ignore);
            
            if (debugMode)
            {
                Debug.Log($"[ObjectGenerator] Checking BoxCollider at {worldBoxCenter}, size: {boxSize}");
            }
        }
        else if (objCollider is SphereCollider sphereCollider)
        {
            float radius = sphereCollider.radius * Mathf.Max(
                objectToSpawn.transform.localScale.x,
                objectToSpawn.transform.localScale.y,
                objectToSpawn.transform.localScale.z
            );
            Vector3 sphereCenter = position + sphereCollider.center;
            Vector3 worldSphereCenter = transform.TransformPoint(sphereCenter);
            hitColliders = Physics.OverlapSphere(worldSphereCenter, radius, -1, QueryTriggerInteraction.Ignore);
            
            if (debugMode)
            {
                Debug.Log($"[ObjectGenerator] Checking SphereCollider at {worldSphereCenter}, radius: {radius}");
            }
        }
        else if (objCollider is CapsuleCollider capsuleCollider)
        {
            float radius = capsuleCollider.radius * Mathf.Max(
                objectToSpawn.transform.localScale.x,
                objectToSpawn.transform.localScale.z
            );
            Vector3 capsuleCenter = position + capsuleCollider.center;
            Vector3 worldCapsuleCenter = transform.TransformPoint(capsuleCenter);
            hitColliders = Physics.OverlapSphere(worldCapsuleCenter, radius, -1, QueryTriggerInteraction.Ignore);
            
            if (debugMode)
            {
                Debug.Log($"[ObjectGenerator] Checking CapsuleCollider at {worldCapsuleCenter}, radius: {radius}");
            }
        }
        else if (objCollider is MeshCollider meshCollider)
        {
            // MeshColliderの場合はBounds（境界ボックス）を使用
            Bounds bounds = meshCollider.sharedMesh != null ? meshCollider.sharedMesh.bounds : meshCollider.bounds;
            Vector3 scaledSize = Vector3.Scale(bounds.size, objectToSpawn.transform.localScale);
            Vector3 boundsCenter = position + bounds.center;
            Vector3 worldBoundsCenter = transform.TransformPoint(boundsCenter);
            hitColliders = Physics.OverlapBox(worldBoundsCenter, scaledSize / 2f, Quaternion.identity, -1, QueryTriggerInteraction.Ignore);
            
            if (debugMode)
            {
                Debug.Log($"[ObjectGenerator] Checking MeshCollider at {worldBoundsCenter}, bounds size: {scaledSize}");
            }
        }
        else
        {
            Debug.LogWarning($"[ObjectGenerator] Unsupported Collider type: {objCollider.GetType().Name}. Using default sphere check.");
            // サポートされていないColliderの場合はデフォルトの球体チェック
            hitColliders = Physics.OverlapSphere(worldPosition, 0.5f, -1, QueryTriggerInteraction.Ignore);
        }

        // 衝突判定結果をチェック（トリガーは既に除外済み）
        if (hitColliders != null)
        {
            foreach (var collider in hitColliders)
            {
                // Prefab自身との誤検出を防ぐ
                if (collider.gameObject == objectToSpawn)
                {
                    if (debugMode)
                    {
                        Debug.Log($"[ObjectGenerator] Ignoring prefab itself: {collider.gameObject.name}");
                    }
                    continue;
                }
                
                // 既に生成されたオブジェクト（このGeneratorの子）との衝突は許可しない
                if (collider.transform.parent == transform)
                {
                    if (debugMode)
                    {
                        Debug.Log($"[ObjectGenerator] Position {position} collides with already spawned object ({collider.GetType().Name}): {collider.gameObject.name}");
                    }
                    return false;
                }
                
                // シーン内の他のオブジェクトとの衝突もチェック
                if (collider.transform.parent != transform)
                {
                    if (debugMode)
                    {
                        Debug.Log($"[ObjectGenerator] Position {position} collides with scene object ({collider.GetType().Name}): {collider.gameObject.name}");
                    }
                    return false;
                }
            }
        }
        
        return true;
    }

}
