using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class EnvEntity : MonoBehaviour
{
    // Attached to each GameObject we will consider as an entity in the physical
    // environment. This class is responsible for doing two things; 1) assigning
    // a unique identifier, 2) computing 2D axis-aligned bounding boxes (AABBs)
    // whenever needed

    // Storage of AABBs, maintained as dictionary from Camera target display id to Rect
    public Dictionary<int, Rect> boxes;

    // List of cameras for each of which AABBs should be computed and stored for
    // the object this component is attached to
    [HideInInspector]
    public List<Camera> cameras;

    // Unique string identifier
    [HideInInspector]
    public string uid;

    private void Awake()
    {
        // Initialize the dictionary for storing AABBs
        boxes = new Dictionary<int, Rect>();

        // Register existing cameras
        cameras = new List<Camera>();
        foreach (var cam in Camera.allCameras)
        {
            cameras.Add(cam);
        }

        // Assign uid by first generating a GUID and taking a prefix
        uid = Guid.NewGuid().ToString()[..8];
    }

    private void Start()
    {
        // ComputeBoxes();
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

    // Call to update the storage of Camera->Rect mapping; may be recursively evoked
    // by a parent EnvEntity, make sure AABB is computed once and only once
    // for the frame!
    public void ComputeBoxes()
    {
        // Can skip if gameObject hasn't moved since
        if (!gameObject.transform.hasChanged) return;

        // Checking if the associated GameObject is an 'atomic' one with no further
        // EnvEntity components on descendants.
        // If so, enumerate over vertices to compute AABB from screen coordinates
        // of the extremities; otherwise, compute from children's AABBs.
        var childrenEntities = ClosestChildren(gameObject);
        var isAtomic = childrenEntities.Count == 0;

        // Compute and update AABBs for each camera
        foreach (var cam in cameras)
        {
            // Initialize locations screen-aligned box extremities to be updated
            var xMin = float.MaxValue; var xMax = float.MinValue;
            var yMin = float.MaxValue; var yMax = float.MinValue;

            if (isAtomic)
            {
                // Iterate through all vertices in the mesh colliders of self and children
                // to obtain the tightest enclosing 2D box aligned w.r.t. screen
                var meshes = gameObject.GetComponentsInChildren<MeshCollider>();
                foreach (var mf in meshes)
                {
                    var vertices = mf.sharedMesh.vertices;
                    foreach (var v in vertices)
                    {
                        // From position in local to world coordinate, then to screen coordinate
                        var worldPoint = gameObject.transform.TransformPoint(v);
                        var screenPoint = cam.WorldToScreenPoint(worldPoint);

                        // Need to flip y position in screen coordinate to comply with MouseMoveEvent
                        screenPoint.y = Screen.height - screenPoint.y;

                        // Update extremities
                        xMin = Math.Min(xMin, screenPoint.x);
                        yMin = Math.Min(yMin, screenPoint.y);
                        xMax = Math.Max(xMax, screenPoint.x);
                        yMax = Math.Max(yMax, screenPoint.y);
                    }
                }
            }
            else
            {
                // Compute from children's boxes
                foreach (var cEnt in childrenEntities)
                {
                    // Recursively call for the child entity to ensure its AABB is computed
                    cEnt.ComputeBoxes();
                    var cBox = cEnt.boxes[cam.targetDisplay];

                    // Update extremities
                    xMin = Math.Min(xMin, cBox.x);
                    yMin = Math.Min(yMin, cBox.y);
                    xMax = Math.Max(xMax, cBox.x+cBox.width);
                    yMax = Math.Max(yMax, cBox.y+cBox.height);
                }
            }

            // Create and assign a new rect representing the box
            boxes[cam.targetDisplay] = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        }

        // Reset this flag to false after computing AABBs, so that they are not computed
        // again until gameObject moves again
        gameObject.transform.hasChanged = false;
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
            var childEntity = tr.gameObject.GetComponent<EnvEntity>();

            // If EnvEntity component not found, recurse on the child gameObject; otherwise,
            // add to list and do not recurse further
            if (childEntity is null)
                closestChildren = closestChildren.Concat(ClosestChildren(tr.gameObject)).ToList();
            else
            {
                // If EnvEntity component disabled, disregard (gameObject likely to be destroyed)
                if (childEntity.enabled) closestChildren.Add(childEntity);
            }
        }

        return closestChildren;
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
}
