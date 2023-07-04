using UnityEngine;

public class EntityRef
{
    public readonly string stringRef;
    public readonly float[] maskRef;
    public readonly EntityRefType refType;

    public EntityRef(float[] reference)
    {
        maskRef = reference;
        refType = EntityRefType.Mask;
    }
    public EntityRef(string reference)
    {
        stringRef = reference;
        refType = EntityRefType.String;
    }
}

public enum EntityRefType { String, Mask }
