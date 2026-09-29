import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  ALL_WORDS,
  AVAILABLE_WORDS,
  CONFUSABLE_GROUPS,
  DECKS,
  TRICKY_ACCURACY_THRESHOLD,
  availableDecks,
  classIdForWord,
  compareCandidates,
  hasDemoVideos,
  heldOutAccuracy,
  isTricky,
  wordForClassId,
  wordsInDeck,
} from "./words";
import classAccuracy from "./class_accuracy.json";

const root = resolve(__dirname, "../..");
const labelMap = JSON.parse(readFileSync(resolve(root, "public/models/ksl_f/label_map.json"), "utf8")).label_map;
const videoFiles = readdirSync(resolve(root, "public/videos"));

/** The 31 practice words (and Korean text) that shipped before the 67-word expansion. */
const ORIGINAL_31: Record<string, string> = {
  hi: "안녕", meet: "만나다", glad: "기쁘다", me: "나", name: "이름", equal: "같다", eat: "먹다", "do effort": "노력하다",
  age: "나이", again: "다시", "how many": "얼마나", day: "날", when: "언제", subway: "지하철", family: "가족",
  please: "부탁하다", sister: "언니/누나", study: "공부하다", human: "사람", now: "지금", end: "끝", you: "당신",
  worried: "걱정하다", marry: "결혼하다", no: "아니요", sweat: "땀", yet: "아직", born: "태어나다", Seoul: "서울",
  dinner: "저녁", food: "음식",
};

describe("vocabulary covers all 67 trained classes", () => {
  it("has exactly the trained class ids, once each, with unique words", () => {
    expect(ALL_WORDS).toHaveLength(67);
    expect(ALL_WORDS.map((w) => w.classId).sort((a, b) => a - b)).toEqual(
      Object.keys(labelMap).map(Number).sort((a, b) => a - b),
    );
    expect(new Set(ALL_WORDS.map((w) => w.english)).size).toBe(67);
  });

  it("keeps the original 31 words and Korean text, and flags exactly the other 36 as unverified", () => {
    for (const [english, korean] of Object.entries(ORIGINAL_31)) {
      const w = ALL_WORDS.find((x) => x.english === english);
      expect(w, english).toBeDefined();
      expect(w!.korean).toBe(korean);
      expect(w!.koreanVerified).toBe(true);
    }
    const unverified = ALL_WORDS.filter((w) => !w.koreanVerified);
    expect(unverified).toHaveLength(36);
    expect(unverified.every((w) => w.korean.trim().length > 0 && !(w.english in ORIGINAL_31))).toBe(true);
  });

  it("resolves words <-> class ids (incl. names that differ from the KSL-77 class labels)", () => {
    expect(classIdForWord("Seoul")).toBe(67);
    expect(classIdForWord("please")).toBe(43);
    expect(classIdForWord("worried")).toBe(57);
    expect(wordForClassId(24)).toBe("nice");
    expect(wordForClassId(57)).toBe("worried");
    expect(wordForClassId(999)).toBe("sign #999");
  });
});

describe("decks", () => {
  it("assigns every word to a defined deck and every deck has words", () => {
    const ids = new Set(DECKS.map((d) => d.id));
    expect(ALL_WORDS.every((w) => ids.has(w.deck))).toBe(true);
    for (const d of DECKS) expect(ALL_WORDS.some((w) => w.deck === d.id), d.id).toBe(true);
  });

  it("deck contents partition the available words", () => {
    const fromDecks = DECKS.flatMap((d) => wordsInDeck(d.id).map((w) => w.english));
    expect(fromDecks.sort()).toEqual(AVAILABLE_WORDS.map((w) => w.english).sort());
    expect(wordsInDeck("all")).toHaveLength(AVAILABLE_WORDS.length);
    expect(availableDecks().reduce((n, d) => n + d.count, 0)).toBe(AVAILABLE_WORDS.length);
  });
});

describe("demo videos", () => {
  it("availability matches the files on disk (both clips, exact-case lowercase .mp4)", () => {
    for (const w of ALL_WORDS) {
      const both = videoFiles.includes(`${w.english}_example1.mp4`) && videoFiles.includes(`${w.english}_example2.mp4`);
      expect(hasDemoVideos(w.english), w.english).toBe(both);
    }
  });
  it("every practiceable word has a trained class (no untrained words are offered)", () => {
    for (const w of AVAILABLE_WORDS) expect(String(w.classId) in labelMap, w.english).toBe(true);
    expect(AVAILABLE_WORDS.length).toBeGreaterThanOrEqual(49);
  });
});

describe("tricky signs and compare groups", () => {
  it("has held-out accuracy for every class, and tags only those under the threshold", () => {
    for (const w of ALL_WORDS) {
      const a = heldOutAccuracy(w.english);
      expect(a, w.english).toBeTypeOf("number");
      expect(isTricky(w.english)).toBe(a! < TRICKY_ACCURACY_THRESHOLD);
    }
    expect(Object.keys(classAccuracy.classes)).toHaveLength(67);
    expect(isTricky("success")).toBe(true); // 52.6% held-out
    expect(isTricky("hi")).toBe(false);
  });

  it("confusable groups only mention real words; candidates are the available mates in the same group", () => {
    const english = new Set(ALL_WORDS.map((w) => w.english));
    for (const g of CONFUSABLE_GROUPS) for (const w of g) expect(english.has(w), w).toBe(true);
    for (const w of ALL_WORDS) {
      const mates = new Set(CONFUSABLE_GROUPS.filter((g) => g.includes(w.english)).flat());
      for (const c of compareCandidates(w.english)) {
        expect(c).not.toBe(w.english);
        expect(mates.has(c)).toBe(true);
        expect(hasDemoVideos(c)).toBe(true);
      }
    }
    expect(compareCandidates("when")).toContain("time"); // time has clips today
    expect(compareCandidates("hi")).toEqual([]);
  });
});
