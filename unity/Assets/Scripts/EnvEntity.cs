using System;
using System.Collections.Generic;
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

    // Start is called before the first frame update
    private void Start()
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

    // Update is called once per frame
    private void Update()
    {
        ComputeBoxes();
    }

    // Call to update the storage of Camera->Rect mapping; may be recursively evoked
    // by a parent EnvEntity, make sure AABB is computed once and only once
    // for the frame!
    private void ComputeBoxes()
    {
        // Can skip if gameObject hasn't moved since
        if (!gameObject.transform.hasChanged) return;

        // Checking if the associated GameObject is an 'atomic' one with MeshFilter.
        // If so, enumerate over vertices to compute AABB from screen coordinates
        // of the extremities; otherwise, compute from children's AABBs.
        var childrenExtractors = new List<EnvEntity>();

        // A Unity peculiarity exploited here is that Transform component of a GameObject
        // implements IEnumerable interface to enumerate its immediate children
        // (Unity doesn't have a built-in method for enumerating immediate children of
        // a GameObject, just (recursive) one for all descendants)
        foreach (Transform tr in gameObject.transform)
        {
            var childExtractor = tr.gameObject.GetComponent<EnvEntity>();

            // Add to list if exists
            if (childExtractor is not null)
                childrenExtractors.Add(childExtractor);
        }
        var isAtomic = childrenExtractors.Count == 0;

        // Compute and update AABBs for each camera
        foreach (var cam in cameras)
        {
            // Initialize locations screen-aligned box extremities to be updated
            var xMin = float.MaxValue; var xMax = float.MinValue;
            var yMin = float.MaxValue; var yMax = float.MinValue;

            if (isAtomic)
            {
                // Iterate through all vertices in the mesh filter to obtain the
                // tightest enclosing 2D box aligned w.r.t. screen
                var vertices = gameObject.GetComponent<MeshFilter>().mesh.vertices;
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
            else
            {
                // Compute from children's boxes
                foreach (var cEnt in childrenExtractors)
                {
                    // Recursively call for the child extractor to ensure its AABB is computed
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

    public static EnvEntity FindByUid(string uid)
    {
        // Fetch EnvEntity with matching uid (if exists)
        var allEnts = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None);
        foreach (var ent in allEnts)
        {
            if (ent.uid == uid) return ent;
        }

        return null;
    }
}
