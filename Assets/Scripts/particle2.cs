using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class particle2 : MonoBehaviour
{
    void Update()
    {
        var particleSystem = GetComponent<ParticleSystem>();
	var main = particleSystem.main;
	main.startSize = Random.Range(0.1f,0.2f);

	var emson = particleSystem.emission;
	emson.rateOverTime = 10f;
    }
}
