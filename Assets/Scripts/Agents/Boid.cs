using UnityEngine;

/// <summary>
/// Boidクラス
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public abstract class Boid : BaseAgent
{
    public float innerRadius = 6.6f;        // 分離領域半径
    public float separatePower = 1.2f;      // 分離力
    public float alignPower = 3.3f;         // 整列力
    public float cohesionPower = 0.6f;      // 結合力
    public float destinationPower = 0.2f;   // 目的地への重視度

    /// <summary>
    /// 他のエージェントから距離をとるメソッド
    /// </summary>
    /// <returns>分離ベクトル</returns>
    public abstract Vector3 Separation();


    /// <summary>
    /// 視界内のエージェントと向きを合わせるメソッド
    /// </summary>
    /// <returns>整列ベクトル</returns>
    public abstract Vector3 Align();

    /// <summary>
    /// 視界内のエージェントの中心座標を目指すメソッド
    /// </summary>
    /// <returns>結合ベクトル</returns>
    public abstract Vector3 Cohesion();

    /// <summary>
    /// エージェント以外のオブジェクトから距離をとるメソッド
    /// </summary>
    /// <returns>回避ベクトル</returns>
    public abstract Vector3 Avoid();

    /// <summary>
    /// エージェントのメインミッションを実行する抽象メソッド
    /// </summary>
    /// <returns></returns>
    public  override abstract Vector3 ExecutedMission();

}
