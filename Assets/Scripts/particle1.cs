using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class particle1 : MonoBehaviour
{

    void Update()
    {
        var particleSystem = GetComponent<ParticleSystem>();
        var main = particleSystem.main;
        main.startSize = Random.Range(0.1f, 0.4f);

        var emson = particleSystem.emission;
        emson.rateOverTime = 250f;
    }
}
