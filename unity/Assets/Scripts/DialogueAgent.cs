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
    public readonly Queue<(string, string, Dictionary<(int, int), EntityRef>)> outgoingMsgBuffer = new();

    // Stores queue of ground-truth mask info requests received from the backend
    public readonly Queue<string> gtMaskRequests = new();

    // Communication side channel to Python backend for requesting decisions
    protected string channelUuid;
    protected MessageSideChannel backendMsgChannel;
    
    // Camera sensor & annotating perception camera component
    private CameraSensorComponent _cameraSensor;
    private PerceptionCamera _perCam;

    // Behavior type as MLAgent
    private BehaviorType _behaviorType;

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
                    var refEnt = EnvEntity.FindByUid(entUid);
                    float[] screenMask;
                    if (refEnt.isBogus)
                    {
                        // Bogus entity, mask directly stores the bitmap
                        screenMask = refEnt.masks[_cameraSensor.Camera.targetDisplay]
                            .Select(c => c.a > 0f ? 1f : 0f).ToArray();
                    }
                    else
                    {
                        // Non-bogus entity, mask stores set of matching colors, which need to be
                        // translated to bitmap
                        
                        // Retrieve referenced EnvEntity and fetch segmentation mask in absolute scale
                        // w.r.t. this agent's camera's target display screen
                        var maskColors = refEnt.masks[_cameraSensor.Camera.targetDisplay];
                        var segMapBuffer = EnvEntity.annotationStorage.segMap;
                        screenMask = ColorsToMask(segMapBuffer, maskColors);
                        
                    }
                    demRefs[range] = new EntityRef(
                        MaskCoordinateSwitch(screenMask, true)
                    );
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
        var messagesToUtter = new List<(string, string, Dictionary<(int, int), EntityRef>)>();
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
            .Select(m => m.Item3)
            .Any(rfs => rfs is not null && rfs.Count > 0);

        // If needed, synchronously wait until masks are updated and captured
        if (masksNeeded)
        {
            yield return StartCoroutine(CaptureAnnotations());
            
            // If any ground-truth mask requests are pending, handle them here by
            // sending info to backend
            if (gtMaskRequests.Count > 0)
            {
                var responseString = "GT mask response: ";
                var responseMasks = new Dictionary<(int, int), EntityRef>();
                var stringPointer = responseString.Length;

                var partStrings = new List<string>();
                while (gtMaskRequests.Count > 0)
                {
                    var req = gtMaskRequests.Dequeue();
                    partStrings.Add(req);

                    var range = (stringPointer, stringPointer+req.Length);
                    
                    // Find relevant EnvEntity and fetch mask
                    var foundEnt = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None)
                        .FirstOrDefault(
                            ent =>
                            {
                                var parent = ent.gameObject.transform.parent;
                                var hasParent = parent is not null;
                                return hasParent && req == parent.gameObject.name;
                            }
                        );

                    if (foundEnt is null) throw new Exception("Invalid part type");

                    var maskColors = foundEnt.masks[_cameraSensor.Camera.targetDisplay];
                    var segMapBuffer = EnvEntity.annotationStorage.segMap;
                    var screenMask = ColorsToMask(segMapBuffer, maskColors);
                    responseMasks[range] = new EntityRef(
                        MaskCoordinateSwitch(screenMask, true)
                    );

                    stringPointer += req.Length;
                    if (gtMaskRequests.Count > 0) stringPointer += 2;   // Account for ", " delimiter
                }
                responseString += string.Join(", ", partStrings.ToArray());

                backendMsgChannel.SendMessageToBackend(
                    "System", responseString, responseMasks
                );
            }
        }

        // Now utter individual messages
        foreach (var (speaker, utterance, demRefs) in messagesToUtter)
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
                            var screenMask = MaskCoordinateSwitch(demRef.maskRef, false);
                            // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
                            demRefsResolved[range] = EnvEntity.FindByMask(screenMask, targetDisplay).uid;
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

    protected IEnumerator CaptureAnnotations()
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_maskCapturing) yield break;
        _maskCapturing = true;

        // First send a request for a capture to the PerceptionCamera component of the
        // Camera to which the CameraSensorComponent is attached
        yield return null;      // Wait for a frame to render
        _perCam.RequestCapture();
        for (var i=0; i < 5; i++)
            yield return null;
        // Waiting several more frames to ensure annotations were captured (This is
        // somewhat ad hoc indeed... but it works)

        // Wait until annotations are ready in the storage endpoint for retrieval
        yield return new WaitUntil(() => EnvEntity.annotationStorage.annotationsUpToDate);

        // Finally, update segmentation masks of all EnvEntity instances based on the data
        // stored in the endpoint
        EnvEntity.UpdateAnnotationsAll();

        // Reset flag on exit
        _maskCapturing = false;
    }

    private static float ContainsColor(Color32 color, Color32[] colorSet)
    {
        // Test if input color is contained in an array of colors
        return colorSet.Any(c => c.r == color.r && c.g == color.g && c.b == color.b) ? 1f : 0f;
    }

    private static float[] ColorsToMask(Color32[] segMapBuffer, Color32[] entities)
    {
        // Convert set of Color32 values to binary mask based on the segmentation image buffer
        var binaryMask = segMapBuffer
            .Select(c => ContainsColor(c, entities)).ToArray();

        return binaryMask;
    }

    private float[] MaskCoordinateSwitch(float[] sourceMask, bool screenToSensor)
    {
        var targetDisplay = _cameraSensor.Camera.targetDisplay;
        var screenWidth = Display.displays[targetDisplay].renderingWidth;
        var screenHeight = Display.displays[targetDisplay].renderingHeight;
        var sensorWidth = _cameraSensor.Width;
        var sensorHeight = _cameraSensor.Height;

        int sourceWidth, sourceHeight, targetWidth, targetHeight;
        if (screenToSensor)
        {
            sourceWidth = screenWidth; sourceHeight = screenHeight;
            targetWidth = sensorWidth; targetHeight = sensorHeight;
        }
        else
        {
            sourceWidth = sensorWidth; sourceHeight = sensorHeight;
            targetWidth = screenWidth; targetHeight = screenHeight;
        }

        // Texture2D representation of the provided mask to be manipulated, in the
        // screen coordinate
        var sourceTexture = new Texture2D(sourceWidth, sourceHeight);
        sourceTexture.SetPixels(
            sourceMask.Select(v => new Color(1f, 1f, 1f, v)).ToArray()
        );

        // To target coordinate, rescale by height ratio
        // (For some reason it's freakishly inconvenient to resize images in Unity)
        var heightRatio = (float) targetHeight / sourceHeight;
        var newWidth = (int) (sourceWidth * heightRatio);
        var rescaledSourceTexture = new Texture2D(newWidth, targetHeight);
        for (var i = 0; i < newWidth; i++)
        {
            var u = (float) i / newWidth;
            for (var j = 0; j < targetHeight; j++)
            {
                var v = (float) j / targetHeight;
                var interpolatedPixel = sourceTexture.GetPixelBilinear(u, v);
                rescaledSourceTexture.SetPixel(i, targetHeight-j, interpolatedPixel);
                    // Flip y-axis
            }
        }
        var rescaledMask = rescaledSourceTexture.GetPixels();

        // X-axis offset for copying over mask data
        var xOffset = (newWidth - targetWidth) / 2;

        // Read values from the rescaled sourceTexture, row by row, to obtain the
        // final mask transformation
        var targetMask = new float[targetWidth * targetHeight];
        for (var j = 0; j < targetHeight; j++)
        {
            int sourceStart, sourceEnd, targetStart;
            if (xOffset > 0)
            {
                // Source is 'wider' in aspect ratio, read with the x-axis offset
                sourceStart = j * newWidth + xOffset;
                sourceEnd = sourceStart + targetWidth;
                targetStart = j * targetWidth;
            }
            else
            {
                // Source is 'narrower' in aspect ratio, write with the x-axis offset
                sourceStart = j * newWidth;
                sourceEnd = sourceStart + newWidth;
                targetStart = j * targetWidth - xOffset;
            }

            for (var i = 0; i < sourceEnd-sourceStart; i++)
                targetMask[i+targetStart] = rescaledMask[i+sourceStart].a;
        }

        // Free them
        Destroy(sourceTexture);
        Destroy(rescaledSourceTexture);

        return targetMask;
    }
}

