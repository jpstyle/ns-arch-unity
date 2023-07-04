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
    // id to binary maps
    public Dictionary<int, float[]> masks;
    // Storage of axis-aligned bounding boxes (AABBs), inferred from the segmentation
    // masks; also dictionary from display id to boxes
    public Dictionary<int, Rect> boxes;

    // Unique string identifier
    [HideInInspector]
    public string uid;

    [HideInInspector]
    // Whether this EnvEntity is 'atomic', having no further EnvEntities as descendants
    public bool isAtomic;

    // Storage endpoint registered as static field so that it can be accessed anywhere
    public static StorageEndpoint annotationStorage;

    // Storage of closest EnvEntity children; may be empty
    private List<EnvEntity> _closestChildren;

    // Boolean flag whether this entity has its up-to-date segmentation masks (per camera)
    // computed and ready
    private bool _masksUpdated;
    
    // List of cameras for each of which masks should be computed and stored for
    // the entity; dictionary mapping from PerceptionCamera ID to target display
    private static Dictionary<string, int> _perCamIDToDisplay;

    private void Awake()
    {
        // Initialize the dictionary for storing masks & boxes
        masks = new Dictionary<int, float[]>();
        boxes = new Dictionary<int, Rect>();

        UpdateClosestChildren();        // Invoked at awake
    }

    private void Start()
    {
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
    
    // Call to update the storage of Camera->float[] (mask) mapping; may be recursively evoked
    // by a parent EnvEntity, make sure mask is computed once and only once for the frame!
    private void UpdateMasks()
    {
        if (_masksUpdated) return;      // Already computed for this frame

        if (isAtomic)
        {
            foreach (var (perCamID, perCamMasks) in annotationStorage.maskStorage)
            {
                var displayId = _perCamIDToDisplay[perCamID];

                if (perCamMasks.TryGetValue(uid, out var msk))
                {
                    // Entity found, being visible from the PerceptionCamera
                    masks[displayId] = msk;
                }
                // Not updated if entirely occluded by some other entity and thus not
                // visible to the PerceptionCamera
            }
        }
        else
        {
            // Iterate through closest children EnvEntity instances to compute the segmentation
            // mask enclosing all of them, per display
            var newMaskPerDisplay = new Dictionary<int, float[]>();
            foreach (var cEnt in _closestChildren)
            {
                // Recursively call for the child entity to ensure its mask is computed
                cEnt.UpdateMasks();

                foreach (var (displayId, msk) in cEnt.masks)
                {
                    if (newMaskPerDisplay.ContainsKey(displayId))
                    {
                        // Compare with current extremities and take min/max to obtain
                        // minimally enclosing mask
                        var currentMask = newMaskPerDisplay[displayId];
                        newMaskPerDisplay[displayId] = 
                            currentMask.Zip(msk, (v1, v2) => v1+v2).ToArray();
                    }
                    else
                    {
                        // First entry, initialize with this mask 
                        newMaskPerDisplay[displayId] = msk;
                    }
                }
            }

            // Now register the discovered minimal segmentation mask; cap values higher
            // than 1.0 at 1.0 ceiling
            foreach (var (displayId, newMask) in newMaskPerDisplay)
                masks[displayId] = newMask.Select(v => Math.Min(v, 1f)).ToArray();
        }

        // Obtain bounding boxes and store as Rect as well; consumed by pointer UI
        foreach (var (displayId, msk) in masks)
        {
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
            boxes[displayId] = new Rect(xMin, yMin, xMax-xMin, yMax-yMin);
        }

        // Set this flag to true after computing masks, so that this method is not invoked
        // again when the masks are already updated 
        _masksUpdated = true;
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

    public void UpdateClosestChildren()
    {
        // Find and store closest EnvEntity children at the time invoked
        _closestChildren = ClosestChildren(gameObject);
        isAtomic = _closestChildren.Count == 0;
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
        // Fetch EnvEntity with highest mask IoU on specified provided display (if exists)
        EnvEntity refEnt = null;
        var maxIoU = 0f;

        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEntities)
        {
            if (!ent.enabled) continue;
            if (!ent.masks.ContainsKey(displayId)) continue;

            var entMask = ent.masks[displayId];
            
            // Intersection & union of two masks
            var maskIntersection = entMask.Zip(msk, (v1, v2) => v1 * v2).ToArray();
            var maskUnion = entMask.Zip(msk, (v1, v2) => v1 + v2)
                .Select(v => Math.Min(v, 1f)).ToArray();

            // Continue if sum of intersection is not larger than zero (i.e., no overlap)
            var overlaps = maskIntersection.Sum() > 0;
            if (!overlaps) continue;

            // Compute mask IoU
            var maskIoU = maskIntersection.Sum() / maskUnion.Sum();
            if (maskIoU > maxIoU)
            {
                maxIoU = maskIoU;
                refEnt = ent;
            }
        }

        return refEnt;
    }

    public static void UpdateMasksAll()
    {
        // Static method for running UpdateMasks for all existing EnvEntity instances

        // Set flags for all instances that their masks are currently being updated 
        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        Array.ForEach(allEntities, e => { e._masksUpdated = false; });

        foreach (var ent in allEntities)
            ent.UpdateMasks();
    }
}
