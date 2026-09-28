import type { Grade } from "./grading";

/** User-facing feedback text for a grade. `wordFor` maps an original class id to a display word. */
export function feedbackMessage(grade: Grade, wordFor: (classId: number) => string, targetClassId?: number): string {
  switch (grade.status) {
    case "correct":
      return "Perfect!";
    case "close":
      // Target was top-1 but under CORRECT_MIN: don't "name" the target as if it were a different word.
      return grade.top1 === targetClassId || grade.top1 === undefined
        ? "Almost — that was close. Try once more, a little more clearly."
        : `Almost — it looked a bit like “${wordFor(grade.top1)}”.`;
    case "confused":
      return grade.top1 !== undefined ? `That looked like “${wordFor(grade.top1)}”.` : "Not quite — watch the example and try again.";
    case "incorrect":
      return "Not quite — watch the example and try again.";
    case "not_detected":
      return grade.reason === "hands"
        ? "I couldn't see your hands much — keep them in view while signing."
        : "I couldn't see you well — make sure your face and shoulders are in view, then try again.";
  }
}
