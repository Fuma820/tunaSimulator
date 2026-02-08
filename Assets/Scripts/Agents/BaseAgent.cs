using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Agent抽象クラス
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public abstract class BaseAgent : MonoBehaviour
{
    public float moveSpeed = 3.1f;                              // 移動速度の制限値
    public Vector3 destination = new(200, 20, 200);             // 目的地座標
    public float rotationSpeed = 100.0f;                        // 回転速度の制限値

    protected Rigidbody rb;
    protected List<BaseAgent> outerList = new();                // 認識したエージェントのリスト
    protected List<GameObject> objectList = new();              // 認識したエージェント以外のオブジェクトxのリスト
    protected Vector3 moveVector;
    public Vector3 GetVelocity { get { return rb.velocity; } }  // 現在の速度ベクトル
    protected Renderer objectRenderer;

    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
        // ランダムな方向を向く
        this.transform.LookAt(Random.onUnitSphere, Vector3.up);

        objectRenderer = GetComponent<Renderer>();
    }

    private void Update()
    {
        moveVector = ExecutedMission();

        // 最大速度を制限
        rb.AddForce(moveVector * moveSpeed - rb.velocity);

        // ターゲット方向に向かうための回転を計算
        Quaternion targetRotation = Quaternion.LookRotation(moveVector);
        transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
    }

    private void OnTriggerEnter(Collider other)
    {
        // エージェントとそれ以外のオブジェクトのリストをそれぞれ作成する．
        if (other.gameObject.TryGetComponent<BaseAgent>(out var agent) && !outerList.Contains(agent))
        {
            outerList.Add(agent);
        }
        else if (!outerList.Contains(agent))
        {
            objectList.Add(other.gameObject);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.TryGetComponent<BaseAgent>(out var agent) && outerList.Contains(agent))
        {
            outerList.Remove(agent);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (objectRenderer != null)
        {
            objectRenderer.material.color = Color.blue;
        }
    }

    /// <summary>
    /// エージェントのメインミッションを実行する抽象メソッド
    /// </summary>
    /// <returns></returns>
    public abstract Vector3 ExecutedMission();

}
