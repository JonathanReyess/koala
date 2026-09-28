import type { LabelMap } from "./model";

/** Practice-list words whose spelling differs from the KSL-77 class name. */
const WORD_ALIASES: Record<string, string> = {
  please: "please?",
  worried: "worried_about",
};

export function wordToClassId(wordToId: Record<string, string>, word: string): number | undefined {
  const key = word in wordToId ? word : WORD_ALIASES[word] ?? word;
  const id = wordToId[key];
  return id === undefined ? undefined : Number(id);
}

/** Human-readable label for an original class id, e.g. 57 -> "worried about". */
export function classIdToWord(wordToId: Record<string, string>, classId: number): string {
  const entry = Object.entries(wordToId).find(([, id]) => Number(id) === classId);
  return entry ? entry[0].replace(/_/g, " ").replace(/\?$/, "") : `sign #${classId}`;
}

export function denseToOriginal(labels: LabelMap): (dense: number) => number {
  return (dense) => labels.reverse_label_map[String(dense)];
}

export function isClassInModel(labels: LabelMap, classId: number): boolean {
  return String(classId) in labels.label_map;
}
