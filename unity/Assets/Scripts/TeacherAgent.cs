using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;

public class TeacherAgent : DialogueAgent
{
    // Keep references to prefabs/materials for random initialization at each episode reset
    [SerializeField]
    private List<GameObject> truckTypes;
    [SerializeField]
    private List<GameObject> cabinTypes;
    [SerializeField]
    private List<GameObject> loadTypes;
    [SerializeField]
    private List<GameObject> centerChassisTypes;
    [SerializeField]
    private List<Material> colors;

    protected override void Awake()
    {
        // Register Python-Agent string communication side channel
        // (Note: This works because we will have only one instance of the agent
        // in the scene ever, but ideally we would want 1 channel per instance,
        // with UUIDs generated on the fly each time an instance is created...)
        channelUuid = "da85d4e0-1b60-4c8a-877d-03af30c446f2";
        backendMsgChannel = new MessageSideChannel(channelUuid, this);
        SideChannelManager.RegisterSideChannel(backendMsgChannel);

        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }

    private void EnvironmentReset()
    {
        // Fetch environment parameters received from python backend
        var envParams = Academy.Instance.EnvironmentParameters;

        // Destroy any existing truck & bogus objects
        var existingTruck = GameObject.Find("truck");
        if (existingTruck is not null)
            Destroy(existingTruck);
        var existingBogus = GameObject.Find("bogus");
        if (existingBogus is not null)
            Destroy(existingBogus);
        // Disable all existing EnvEntity components; needed because the Destroy call
        // above is delayed
        foreach (var ent in FindObjectsByType<EnvEntity>(FindObjectsSortMode.None))
            ent.enabled = false;
        // Disable all existing RigidBody components; ditto
        foreach (var rb in FindObjectsByType<Rigidbody>(FindObjectsSortMode.None))
            rb.detectCollisions = false;

        // Sample truck configs
        var truckType = truckTypes[Random.Range(0, truckTypes.Count)];
        var cabinType = cabinTypes[
            (int) envParams.GetWithDefault("cabin_type", Random.Range(0, cabinTypes.Count))
        ];
        var loadType = loadTypes[
            (int) envParams.GetWithDefault("load_type", Random.Range(0, loadTypes.Count))
        ];
        var centerChassisType = centerChassisTypes[Random.Range(0, centerChassisTypes.Count)];

        // Sample part colorings; parts are logically grouped for coloring, indicated
        // by string name prefixes
        var partColorGroups = new Dictionary<string, Material>
        {
            ["chassis_center"] = colors[Random.Range(0, colors.Count)],
            ["cabin"] = colors[Random.Range(0, colors.Count)],
            ["fender"] = colors[Random.Range(0, colors.Count)],
        };

        // Certain combinations of load & center chassis collide and should be rejected
        var combosToReject = new List<(string, string)>
        {
            ("load_dumper", "chassis_center_spares"),
            ("load_dumper", "chassis_center_staircase_4seats"),
            ("load_dumper", "chassis_center_staircase_oshkosh"),
            ("load_rocketLauncher", "chassis_center_spares")
        };
        while (true)        // Rejection sampling
        {
            if (!combosToReject.Contains((loadType.name, centerChassisType.name)))
                break;

            loadType = loadTypes[(int) envParams.GetWithDefault("load_type", 0f)];
            centerChassisType = centerChassisTypes[Random.Range(0, centerChassisTypes.Count)];
        }

        // Instantiate selected truck type
        var truck = Instantiate(truckType);
        truck.name = "truck";

        // Replace truck part prefabs
        var sampledPartsWithHandles = new List<(string, GameObject)>
        {
            ("cabin", cabinType), ("load", loadType), ("chassis_center", centerChassisType)
        };
        foreach (var (partType, sampledPart) in sampledPartsWithHandles)
        {
            var partSlotTf = truck.transform.Find(partType);
            foreach (Transform child in partSlotTf)
            {
                var replacedGObj = child.gameObject;
                replacedGObj.SetActive(false);         // Needed for UpdateClosestChildren below
                Destroy(replacedGObj);
            }
            var newPart = Instantiate(sampledPart, partSlotTf);
            newPart.name = sampledPart.name;        // Not like 'necessary' but just coz
        }
        // Need to update closest children after the replacements
        truck.GetComponent<EnvEntity>().UpdateClosestChildren();

        // Color parts if applicable
        foreach (Transform partSlotTf in truck.transform)
        {
            var partType = partSlotTf.gameObject.name;
            var matchingColorGroups =
                partColorGroups.Where(kv => partType.StartsWith(kv.Key)).ToList();
            if (matchingColorGroups.Count == 0) continue;

            foreach (var mesh in partSlotTf.gameObject.GetComponentsInChildren<MeshRenderer>())
            {
                // Change material only if current one is Default one
                if (mesh.material.name.StartsWith("Default"))
                    mesh.material = matchingColorGroups[0].Value;
            }
        }

        // Random initialization of truck pose
        truck.transform.position = new Vector3(
            Random.Range(-0.25f, 0.25f), 0.85f, Random.Range(0.3f, 0.35f)
        );
        truck.transform.eulerAngles = new Vector3(
            0f, Random.Range(0f, 359.9f), 0f
        );

        // Fast-forward physics simulation until truck rests on the desktop
        Physics.simulationMode = SimulationMode.Script;
        for (var i=0; i<2000; i++)
        {
            Physics.Simulate(Time.fixedDeltaTime);
        }
        Physics.simulationMode = SimulationMode.FixedUpdate;

        // Currently stored annotation info is now obsolete
        EnvEntity.annotationStorage.annotationsUpToDate = false;

        // Clean dialogue history and add new episode header record
        dialogueUI.ClearHistory();
        dialogueUI.CommitUtterance("System", $"Start episode {Academy.Instance.EpisodeCount}");
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (actionBuffers.DiscreteActions[0] == 1)
        {
            // 'Utter' action
            StartCoroutine(Utter());
        }
    }

    public override void Heuristic(in ActionBuffers actionBuffers)
    {
        // Update annotation whenever needed
        if (!EnvEntity.annotationStorage.annotationsUpToDate)
            StartCoroutine(CaptureAnnotations());

        // 'Utter' any outgoing messages
        if (outgoingMsgBuffer.Count > 0)
        {
            var discreteActionBuffers = actionBuffers.DiscreteActions;
            discreteActionBuffers[0] = 1;      // 'Utter'
        }
    }
}
