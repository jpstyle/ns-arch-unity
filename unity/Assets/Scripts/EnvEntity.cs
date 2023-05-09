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
    // a unique identifier, 2) computing 2D axis-aligned bounding boxes (AABBs)
    // whenever needed

    // Storage of AABBs, maintained as dictionary from target display id to Rect
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

    // Boolean flag whether this entity has its up-to-date bounding boxes (per camera)
    // computed and ready
    private bool _boxesUpdated;
    
    // List of cameras for each of which AABBs should be computed and stored for
    // the entity; dictionary mapping from PerceptionCamera ID to target display
    private static Dictionary<string, int> _perCamIDToDisplay;

    private void Awake()
    {
        // Initialize the dictionary for storing AABBs
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
    
    // Call to update the storage of Camera->Rect mapping; may be recursively evoked
    // by a parent EnvEntity, make sure AABB is computed once and only once for the frame!
    private void UpdateBoxes()
    {
        if (_boxesUpdated) return;      // Already computed for this frame

        if (isAtomic)
        {
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
                // Not updated if entirely occluded by some other entity and thus not
                // visible to the PerceptionCamera
            }
        }
        else
        {
            // Iterate through closest children EnvEntity instances to compute the minimal
            // bounding box enclosing all of them, per display
            var extremitiesPerDisplay = new Dictionary<int, (float, float, float, float)>();
            foreach (var cEnt in _closestChildren)
            {
                // Recursively call for the child entity to ensure its AABB is computed
                cEnt.UpdateBoxes();

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

            // Now register the discovered minimal bounding box
            foreach (var (displayId, extremities) in extremitiesPerDisplay)
            {
                var (xMin, yMin, xMax, yMax) = extremities;
                boxes[displayId] = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
            }
        }

        // Set this flag to true after computing AABBs, so that this method is not invoked
        // again when the boxes are already updated 
        _boxesUpdated = true;
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

    public static EnvEntity FindByBox(Rect box, int displayId)
    {
        // Fetch EnvEntity with highest box IoU on specified provided display (if exists)
        EnvEntity refEnt = null;
        var maxIoU = 0f;

        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEntities)
        {
            if (!ent.enabled) continue;
            if (!ent.boxes.ContainsKey(displayId)) continue;

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

        return refEnt;
    }

    public static void UpdateBoxesAll()
    {
        // Static method for running UpdateBoxes for all existing EnvEntity instances

        // Set flags for all instances that their boxes are currently being updated 
        var allEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        Array.ForEach(allEntities, e => { e._boxesUpdated = false; });

        foreach (var ent in allEntities)
            ent.UpdateBoxes();
    }
}
