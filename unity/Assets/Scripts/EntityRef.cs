using UnityEngine;

public class EntityRef
{
    public readonly string stringRef;
    public readonly Rect bboxRef;
    public readonly EntityRefType refType;

    public EntityRef(Rect reference)
    {
        bboxRef = reference;
        refType = EntityRefType.BBox;
    }
    public EntityRef(string reference)
    {
        stringRef = reference;
        refType = EntityRefType.String;
    }
}

public enum EntityRefType { String, BBox }
