using System.Collections.ObjectModel;
using UnityEngine;

public class RecordData : ScriptableObject
{
    public string speaker;
    public string utterance;
    public ReadOnlyDictionary<(int, int), string> demonstrativeReferences;
}
