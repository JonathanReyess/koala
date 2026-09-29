import type { LabelMap } from "./model";

export function denseToOriginal(labels: LabelMap): (dense: number) => number {
  return (dense) => labels.reverse_label_map[String(dense)];
}

export function isClassInModel(labels: LabelMap, classId: number): boolean {
  return String(classId) in labels.label_map;
}
