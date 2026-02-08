using UnityEngine;

public class FogController : MonoBehaviour
{
    public float r = 0.15f;
    public float g = 0.4f;
    public float b = 0.55f;
    public float density = 0.4f;

    void Start()
    {
        RenderSettings.fog = true;

        RenderSettings.fogColor = new Color(r, g, b, 1.0f);

        RenderSettings.fogMode = FogMode.Exponential;

        RenderSettings.fogDensity = density;
    }
}
