using System.Collections.Generic;
using UnityEngine;

public class RecordData : ScriptableObject
{
    public string speaker;
    public string utterance;
    public Dictionary<(int, int), string> DemonstrativeReferences;
}
