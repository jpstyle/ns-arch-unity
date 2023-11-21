using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.LabelManagement;

public class EnvEntity : MonoBehaviour
{
    // Attached to each GameObject we will consider as an entity in the physical
    // environment. This class is responsible for doing two things; 1) assigning
    // a unique identifier, 2) obtaining instance segmentation masks whenever needed

    // Storage of segmentation masks, maintained as dictionary from target display
    // id to corresponding color32
    public Dictionary<int, Color32[]> masks;
    // Storage of axis-aligned bounding boxes (AABBs), inferred from the segmentation
    // masks; also dictionary from display id to boxes
    public Dictionary<int, Rect> boxes;

    // Unique string identifier
    [HideInInspector]
    public string uid;

    [HideInInspector]
    // Whether this EnvEntity is 'atomic', having no further EnvEntities as descendants
    public bool isAtomic;

    [HideInInspector]
    // Whether this EnvEntity is programmatically generated, representing 'bogus' entities
    // detected by some vision module
    public bool isBogus;

    // Storage endpoint registered as static field so that it can be accessed anywhere
    public static StorageEndpoint annotationStorage;

    // Storage of closest EnvEntity children; may be empty
    private List<EnvEntity> _closestChildren;

    // Boolean flag whether this entity has its up-to-date segmentation masks (per camera)
    // computed and ready
    private bool _annotationsUpdated;
    
    // List of cameras for each of which masks should be computed and stored for
    // the entity; dictionary mapping from PerceptionCamera ID to target display
    private static Dictionary<string, int> _perCamIDToDisplay;

    private void Awake()
    {
        if (isBogus) return;

        // Initialize the dictionary for storing masks & boxes
        masks = new Dictionary<int, Color32[]>();
        boxes = new Dictionary<int, Rect>();

        UpdateClosestChildren();        // Invoked at awake
    }

    private void Start()
    {
        if (isBogus) return;

        // Assign uid according to the Labeling component associated with the gameObject
        var labeling = GetComponent<Labeling>();
        if (labeling is null)
            throw new Exception(
                "Associated gameObject of an EnvEntity must have a Labeling component"
            );
        uid = $"ent_{labeling.instanceId}";

        // Register this EnvEntity to existing pointer UI controllers
        var pointerUIs = FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.AddEnvEntity(this);
    }

    private void OnDestroy()
    {
        var pointerUIs = FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.DelEnvEntity(this);
    }

    private static List<EnvEntity> ClosestChildren(GameObject gObj)
    {
        // Find 'closest' children EnvEntity instances, in the sense that the recursive
        // search frontier stops expanding as soon as another EnvEntity instance is
        // encountered; result may be empty
        var closestChildren = new List<EnvEntity>();
        
        // A Unity peculiarity exploited here is that Transform component of a GameObject
        // implements IEnumerable interface to enumerate its immediate children
        // (Unity doesn't have a built-in method for enumerating immediate children of
        // a GameObject, just (recursive) one for all descendants)
        foreach (Transform tr in gObj.transform)
        {
            var childGObj = tr.gameObject;
            if (!childGObj.activeInHierarchy) continue;     // Disregard inactive gameObjects

            var childEntity = childGObj.GetComponent<EnvEntity>();
            if (childEntity is null)
                // If EnvEntity component not found, recurse on the child gameObject; otherwise,
                // add to list and do not recurse further
                closestChildren = closestChildren.Concat(ClosestChildren(tr.gameObject)).ToList();
            else
            {
                // If EnvEntity component disabled, disregard (gameObject likely to be destroyed)
                if (childEntity.enabled) closestChildren.Add(childEntity);
            }
        }

        return closestChildren;
    }
    
    // Call to update the storage of Camera->annotation (mask or box) mapping; may be
    // recursively evoked by a parent EnvEntity, make sure annotations are computed once
    // and only once for the frame!
    private void UpdateAnnotations()
    {
        if (isBogus) return;                   // Bogus entities have static annotations
        if (_annotationsUpdated) return;        // Already computed for this frame

        if (isAtomic)
        {
            // Not updated if entirely occluded by some other entity and thus not
            // visible to the PerceptionCamera
            foreach (var (perCamID, perCamMasks) in annotationStorage.maskStorage)
            {
                var displayId = _perCamIDToDisplay[perCamID];

                if (perCamMasks.TryGetValue(uid, out var msk))
                {
                    // Entity found, being visible from the PerceptionCamera
                    masks[displayId] = new[] {msk};
                }
            }
            foreach (var (perCamID, perCamBoxes) in annotationStorage.boxStorage)
            {
                var targetDisplay = _perCamIDToDisplay[perCamID];

                if (perCamBoxes.TryGetValue(uid, out var box))
                {
                    // Entity found, being visible from the PerceptionCamera
                    boxes[targetDisplay] = new Rect(
                        box.origin.x, box.origin.y,box.dimension.x, box.dimension.y
                    );
                }
            }
        }
        else
        {
            // Iterate through closest children EnvEntity instances to compute the corresponding
            // annotations, per display
            var extremitiesPerDisplay = new Dictionary<int, (float, float, float, float)>();
            var newMaskPerDisplay = new Dictionary<int,Color32[]>();

            foreach (var cEnt in _closestChildren)
            {
                // Recursively call for the child entity to ensure its mask is computed
                cEnt.UpdateAnnotations();

                // Computing masks; simply merge arrays of Color32s
                foreach (var (displayId, msk) in cEnt.masks)
                {
                    if (newMaskPerDisplay.ContainsKey(displayId))
                    {
                        // Merge Color32 lists
                        var currentMask = newMaskPerDisplay[displayId];
                        newMaskPerDisplay[displayId] = currentMask.Concat(msk).ToArray();
                    }
                    else
                    {
                        // First entry, initialize with this Color32 list 
                        newMaskPerDisplay[displayId] = msk;
                    }
                }
                // Computing boxes; find the minimal bounding box enclosing all boxes
                foreach (var (displayId, rect) in cEnt.boxes)
                {
                    if (extremitiesPerDisplay.ContainsKey(displayId))
                    {
                        // Compare with current extremities and take min/max to obtain
                        // minimally enclosing rect
                        var currentExtremities = extremitiesPerDisplay[displayId];
                        extremitiesPerDisplay[displayId] = (
                            Math.Min(currentExtremities.Item1, rect.x),
                            Math.Min(currentExtremities.Item2, rect.y),
                            Math.Max(currentExtremities.Item3, rect.x+rect.width),
                            Math.Max(currentExtremities.Item4, rect.y+rect.height)
                        );
                    }
                    else
                    {
                        // First entry, initialize extremity values with this 
                        extremitiesPerDisplay[displayId] = (
                            rect.x, rect.y, rect.x+rect.width, rect.y+rect.height
                        ); 
                    }
                }
            }
            
            // Now register the discovered masks and boxes
            foreach (var (displayId, colors) in newMaskPerDisplay)
                masks[displayId] = colors;
            foreach (var (displayId, extremities) in extremitiesPerDisplay)
            {
                var (xMin, yMin, xMax, yMax) = extremities;
                boxes[displayId] = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
            }
        }

        // Set this flag to true after computing masks, so that this method is not invoked
        // again when the masks are already updated 
        _annotationsUpdated = true;
    }

    public void UpdateClosestChildren()
    {
        // Find and store closest EnvEntity children at the time invoked
        _closestChildren = ClosestChildren(gameObject);
        isAtomic = _closestChildren.Count == 0;
    }
    
    // Initialize static fields (in Unity, this is preferred rather than using the standard
    // C# static constructors)
    [RuntimeInitializeOnLoadMethod]
    private static void StaticInitialize()
    {
        // Get reference to the consumer endpoint
        annotationStorage = (StorageEndpoint) DatasetCapture.activateEndpoint;

        // Find mapping from PerceptionCamera Ids to the camera's target display
        _perCamIDToDisplay = new Dictionary<string, int>();
        foreach (var cam in Camera.allCameras)
        {
            var perCam = cam.GetComponent<PerceptionCamera>();
            if (perCam is not null)
                _perCamIDToDisplay[perCam.id] = cam.targetDisplay;
        }
    }

    public static EnvEntity FindByObjectPath(string path)
    {
        var pathSplit = path.Split("/");
        Assert.AreEqual(pathSplit[0], "", "Provide full hierarchy path from root");

        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEntities)
        {
            var traverser = ent.transform;
            var match = true;

            for (var i = pathSplit.Length-1; i>=1; i--)
            {
                if (traverser is not null && traverser.name == pathSplit[i])
                {
                    traverser = traverser.parent;
                }
                else
                {
                    match = false;
                    break;
                }
            }

            if (match) return ent;
        }

        return null;
    }
    
    public static EnvEntity FindByUid(string uid)
    {
        // Fetch EnvEntity with matching uid (if exists)
        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEntities)
        {
            if (ent.uid == uid) return ent;
        }

        return null;
    }

    public static EnvEntity FindByMask(float[] msk, int displayId)
    {
        // Fetch EnvEntity with highest box-IoU on specified provided display (if exists).
        // Ideally we would use mask-IoU but it takes too long and box-IoU is sufficiently
        // accurate for our use cases.
        EnvEntity refEnt = null;
        var maxIoU = 0f;

        // Compute tightly enclosing bounding box from provided mask
        var screenWidth = Display.displays[displayId].renderingWidth;
        var screenHeight = Display.displays[displayId].renderingHeight;
        var nonzeroPixels = msk
            .Select((v, i) => (v, i))
            .Where(v => v.Item1 > 0)
            .Select(v => (v.Item2 % screenWidth, screenHeight - v.Item2 / screenWidth))
            .ToArray();

        var xMin = nonzeroPixels.Select(v => v.Item1).Min();
        var xMax = nonzeroPixels.Select(v => v.Item1).Max();
        var yMin = nonzeroPixels.Select(v => v.Item2).Min();
        var yMax = nonzeroPixels.Select(v => v.Item2).Max();
        var box = new Rect(xMin, yMin, xMax-xMin, yMax-yMin);

        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEntities)
        {
            if (!ent.enabled) continue;
            if (!ent.masks.ContainsKey(displayId)) continue;

            var entBox = ent.boxes[displayId];
            if (!entBox.Overlaps(box)) continue;

            // Compute box intersection then IoU
            var intersectionX1 = Math.Max(entBox.x, box.x);
            var intersectionY1 = Math.Max(entBox.y, box.y);
            var intersectionX2 = Math.Min(entBox.x+entBox.width, box.x+box.width);
            var intersectionY2 = Math.Min(entBox.y+entBox.height, box.y+box.height);
            var intersection = new Rect(
                intersectionX1, intersectionY1,
                intersectionX2 - intersectionX1, intersectionY2 - intersectionY1
            );

            var entBoxArea = entBox.width * entBox.height;
            var boxArea = box.width * box.height;
            var intersectionArea = intersection.width * intersection.height;

            var boxIoU = intersectionArea / (entBoxArea+boxArea-intersectionArea);
            if (boxIoU > maxIoU)
            {
                maxIoU = boxIoU;
                refEnt = ent;
            }
        }

        // If provided mask doesn't match any existing EnvEntity above threshold, the mask
        // represents an object instance (possibly not corresponding to any existing Unity
        // scene objects); create a new bogus EnvEntity and manually fill in the mask and
        // box annotation info
        if (maxIoU < 0.9f)
        {
            // Utilizing the max IoU as unique ID...
            var newUid = (int)(maxIoU * 1e6);
            
            // Create a new GameObject and EnvEntity component 
            var bogusObj = new GameObject("bogus", typeof(EnvEntity));
            // ReSharper disable once Unity.PerformanceCriticalCodeInvocation (will be invoked
            // only once for this entity)
            var bogusEnt = bogusObj.GetComponent<EnvEntity>();

            // Initialize appropriately; substituting corresponding pieces of code in Awake()
            // Start(), and UpdateAnnotations() methods
            bogusEnt.isBogus = true;
            bogusEnt.uid = $"ent_{newUid}";
            bogusEnt.masks = new Dictionary<int, Color32[]>();
            bogusEnt.boxes = new Dictionary<int, Rect>();

            // Register the bogus to existing pointer UI controllers
            var pointerUIs = FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
            foreach (var pUI in pointerUIs)
                pUI.AddEnvEntity(bogusEnt);

            // Obtain & store mask and box
            var hotBit = new Color32(0, 0, 0, 255);
            var coldBit = new Color32(0, 0, 0, 0);
            var bogusMask = new Color32[msk.Length];
            var bogusBox = new[]
            {
                float.MaxValue, float.MaxValue, float.MinValue, float.MinValue
            };          // Box extremities (x1, y1, x2, y2)
            for (var i = 0; i < msk.Length; i++)
            {
                var v = msk[i];
                var x = i % screenWidth;
                var y = screenHeight - i / screenWidth;

                if (v > 0f)
                {
                    bogusMask[i] = hotBit;
                    bogusBox[0] = Math.Min(bogusBox[0], x);
                    bogusBox[1] = Math.Min(bogusBox[1], y);
                    bogusBox[2] = Math.Max(bogusBox[2], x);
                    bogusBox[3] = Math.Max(bogusBox[3], y);
                }
                else
                {
                    bogusMask[i] = coldBit;
                }
            }
            bogusEnt.masks[displayId] = bogusMask;
            bogusEnt.boxes[displayId] = new Rect(
                bogusBox[0], bogusBox[1],
                bogusBox[2]-bogusBox[0], bogusBox[3]-bogusBox[1]
            );

            refEnt = bogusEnt;       // To return
        }

        return refEnt;
    }

    public static void UpdateAnnotationsAll()
    {
        // Static method for running UpdateAnnotations for all existing EnvEntity instances

        // Set flags for all instances that their masks are currently being updated 
        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        Array.ForEach(allEntities, e => { e._annotationsUpdated = false; });

        foreach (var ent in allEntities)
            ent.UpdateAnnotations();
    }
}
