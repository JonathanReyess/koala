import type { Grade } from "./grading";

/** User-facing feedback text for a grade. `wordFor` maps an original class id to a display word. */
export function feedbackMessage(grade: Grade, wordFor: (classId: number) => string): string {
  switch (grade.status) {
    case "correct":
      return "Perfect!";
    case "close":
      // Only name another word when the guess is confident (grade.namesTop1 = top-1 != target && p >= CONFUSION_MIN).
      return grade.namesTop1 && grade.top1 !== undefined
        ? `Almost — it looked a bit like “${wordFor(grade.top1)}”.`
        : "Almost — that was close. Try once more.";
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
