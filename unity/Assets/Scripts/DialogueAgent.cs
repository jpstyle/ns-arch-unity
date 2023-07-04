using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

public class DialogueAgent : Agent
// Communication with dialogue UI and communication with Python backend may be
// decoupled later?
{
    // String identifier name for the dialogue participant agent
    public string dialogueParticipantID;

    // Commit utterances through this dialogue UI
    [HideInInspector]
    public DialogueUI dialogueUI;

    // Message communication buffer queues
    public readonly Queue<RecordData> incomingMsgBuffer = new();
    public readonly Queue<(string, Dictionary<(int, int), EntityRef>)> outgoingMsgBuffer = new();

    // Communication side channel to Python backend for requesting decisions
    protected string channelUuid;
    protected MessageSideChannel backendMsgChannel;
    
    // Camera sensor & annotating perception camera component
    private CameraSensorComponent _cameraSensor;
    private PerceptionCamera _perCam;

    // Behavior type as MLAgent
    private BehaviorType _behaviorType;

    // // Only for use by non-Heuristics agents (i.e. connected to Python backend);
    // // boolean flag for checking whether the agent has some unresolved goal to
    // // achieve and thus needs RequestDecision() call
    // protected bool HasGoalToResolve = false;

    // Boolean flag indicating an Utter action is invoked as coroutine and currently
    // running; for preventing multiple invocation of Utter coroutine 
    private bool _uttering;
    
    // Analogous flag for CaptureMask method
    private bool _maskCapturing;

    // For controlling minimal update interval, to allow visual inspection during runs
    private float _nextTimeToAct;
    private const float TimeInterval = 1f;

    public void Start()
    {
        _cameraSensor = GetComponent<CameraSensorComponent>();
        _perCam = _cameraSensor.Camera.GetComponent<PerceptionCamera>();
        if (_perCam is null)
            throw new Exception(
                "This agent's camera sensor doesn't have a PerceptionCamera component"
            );

        _behaviorType = GetComponent<BehaviorParameters>().BehaviorType;
        _nextTimeToAct = Time.time;
    }

    public override void OnEpisodeBegin()
    {
        // Say anything this agent has to say
        StartCoroutine(Utter());
    }

    public void Update()
    {
        if (Time.time < _nextTimeToAct) return;
        
        _nextTimeToAct += TimeInterval;
        if (_behaviorType == BehaviorType.HeuristicOnly)
        {
            // Always empty incoming message buffer and call RequestDecision
            incomingMsgBuffer.Clear();
            RequestDecision();
        }
        else
        {
            // Trying to consult backend for requesting decision only when needed, in order
            // to minimize communication of visual observation data
            if (incomingMsgBuffer.Count == 0) return;

            // Unprocessed incoming messages exist; process and consult backend
            while (incomingMsgBuffer.Count > 0)
            {
                // Fetch single message record from queue
                var incomingMessage = incomingMsgBuffer.Dequeue();
                
                // (If any) Translate EnvEntity reference by UID to segmentation mask w.r.t.
                // this agent's camera sensor
                var demRefs = new Dictionary<(int, int), EntityRef>();
                foreach (var (range, entUid) in incomingMessage.demonstrativeReferences)
                {
                    // Retrieve referenced EnvEntity and fetch segmentation mask in absolute scale
                    // w.r.t. this agent's camera's target display screen
                    var refEnt = EnvEntity.FindByUid(entUid);
                    var screenMask = refEnt.masks[_cameraSensor.Camera.targetDisplay];
                    demRefs[range] = new EntityRef(MaskCoordinateSwitch(screenMask, true));
                }

                // Send message via side channel
                backendMsgChannel.SendMessageToBackend(
                    incomingMessage.speaker, incomingMessage.utterance, demRefs
                );
            }

            // Now wait for decision
            RequestDecision();
        }
    }

    protected IEnumerator Utter()
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_uttering) yield break;
        _uttering = true;

        // Dequeue all messages to utter
        var messagesToUtter = new List<(string, Dictionary<(int, int), EntityRef>)>();
        while (outgoingMsgBuffer.Count > 0)
            messagesToUtter.Add(outgoingMsgBuffer.Dequeue());
        
        // If no outgoing message to utter, can terminate here
        if (messagesToUtter.Count == 0)
        {
            _uttering = false;
            yield break;
        }

        // Check if any of the messages has non-empty demonstrative references and
        // thus segmentation masks need to be captured
        var masksNeeded = messagesToUtter
            .Select(m => m.Item2)
            .Any(rfs => rfs is not null && rfs.Count > 0);

        // If needed, synchronously wait until masks are updated and captured
        if (masksNeeded)
            yield return StartCoroutine(CaptureMasks());

        // Now utter individual messages
        foreach (var (utterance, demRefs) in messagesToUtter)
        {
            if (demRefs is not null && demRefs.Count > 0)
            {
                // Need to resolve demonstrative reference masks to corresponding EnvEntity (uid)
                var demRefsResolved = new Dictionary<(int, int), string>();
                var targetDisplay = _cameraSensor.Camera.targetDisplay;

                foreach (var (range, demRef) in demRefs)
                {
                    switch (demRef.refType)
                    {
                        case EntityRefType.Mask:
                            var screenAbsMask = MaskCoordinateSwitch(demRef.maskRef, false);
                            demRefsResolved[range] = EnvEntity.FindByMask(screenAbsMask, targetDisplay).uid;
                            break;
                        case EntityRefType.String:
                            demRefsResolved[range] = EnvEntity.FindByObjectPath(demRef.stringRef).uid;
                            break;
                        default:
                            // Shouldn't reach here but anyways
                            throw new Exception("Invalid reference data type?");
                    }
                }

                dialogueUI.CommitUtterance(dialogueParticipantID, utterance, demRefsResolved);
            }
            else
                // No demonstrative references to process and resolve
                dialogueUI.CommitUtterance(dialogueParticipantID, utterance);
        }

        // Reset flag on exit
        _uttering = false;
    }

    protected IEnumerator CaptureMasks()
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_maskCapturing) yield break;
        _maskCapturing = true;

        // First send a request for a capture to the PerceptionCamera component of the
        // Camera to which the CameraSensorComponent is attached
        yield return null;              // Wait for a frame to render
        _perCam.RequestCapture();

        // Wait until annotations are ready in the storage endpoint for retrieval
        yield return new WaitUntil(() => EnvEntity.annotationStorage.annotationsUpToDate);

        // Finally, update segmentation masks of all EnvEntity instances based on the data
        // stored in the endpoint
        EnvEntity.UpdateMasksAll();

        // Reset flag on exit
        _maskCapturing = false;
    }

    private float[] MaskCoordinateSwitch(float[] screenMask, bool screenToSensor)
    {
        var targetDisplay = _cameraSensor.Camera.targetDisplay;
        var screenWidth = Display.displays[targetDisplay].renderingWidth;
        var screenHeight = Display.displays[targetDisplay].renderingHeight;

        // Texture2D representation of the provided mask to be manipulated, in the
        // screen coordinate
        var screenTexture = new Texture2D(
            screenWidth, screenHeight, TextureFormat.RHalf, false
        );
        screenTexture.SetPixelData(screenMask, 0);

        // To CameraSensor-relative coordinate; resize by height ratio
        var heightRatio = _cameraSensor.Height / screenHeight;
        var newWidth = screenWidth * heightRatio;
        screenTexture.Reinitialize(newWidth, _cameraSensor.Height);
        var resizedMask = screenTexture.GetPixelData<float>(0).ToArray();

        // X-axis offset for copying over mask data
        var xOffset = (newWidth - _cameraSensor.Width) / 2;

        // New Texture2D representation of the mask in the sensor coordinate; read
        // values from the resized screenTexture, row by row 
        var sensorTexture = new Texture2D(
            _cameraSensor.Width, _cameraSensor.Height, TextureFormat.RHalf, false
        );
        for (var i = 0; i < _cameraSensor.Height; i++)
        {
            int sourceStart, sourceEnd, targetStart;
            if (xOffset > 0)
            {
                // Screen is 'wider' in aspect ratio, read with the x-axis offset
                sourceStart = i * _cameraSensor.Width + xOffset;
                sourceEnd = sourceStart + _cameraSensor.Width;
                targetStart = i * _cameraSensor.Width;
            }
            else
            {
                // Screen is 'narrower' in aspect ratio, write with the x-axis offset
                sourceStart = i * _cameraSensor.Width;
                sourceEnd = sourceStart + _cameraSensor.Width;
                targetStart = i * _cameraSensor.Width - xOffset;
            }

            var rowData = resizedMask[sourceStart..sourceEnd];
            sensorTexture.SetPixelData(rowData, 0, targetStart);
        }

        return sensorTexture.GetPixelData<float>(0).ToArray();
    }
}
