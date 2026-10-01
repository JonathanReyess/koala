import { useState, useEffect, useCallback, useMemo, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Stage } from "@/components/stage/Stage";
import { AppDialog, HeaderChip, HeaderSelect, PillButton, PillButtonRow } from "@/components/ds";
import { CompareDialog } from "@/components/CompareDialog";
import {
  calculateNextReview,
  newWordProgress,
  sanitizeProgress,
  type WordProgress,
} from "@/lib/spacedRepetition";
import {
  ALL_WORDS,
  AVAILABLE_WORDS,
  availableDecks,
  compareCandidates,
  heldOutAccuracy,
  isDeckChoice,
  isTricky,
  wordEntry,
  wordsInDeck,
  type DeckChoice,
} from "@/data/words";
import { RotateCcw, Shuffle, ChevronLeft, ChevronRight, GitCompare, Flame } from "lucide-react";

// =============================================================================
// Constants
// =============================================================================

const MASTERY_CORRECT_THRESHOLD = 3;
const MASTERY_INTERVAL_DAYS = 7;

// Words, decks and Korean text live in src/data (vocab.json + words.ts). Only words whose two demo
// clips exist in public/videos are offered (AVAILABLE_WORDS).
export const getWords = () => AVAILABLE_WORDS.map((w) => w.english);

const DECK_STORAGE_KEY = "koala.deck";

const loadDeckChoice = (): DeckChoice => {
  try {
    const saved = localStorage.getItem(DECK_STORAGE_KEY);
    if (isDeckChoice(saved) && wordsInDeck(saved).length > 0) return saved;
  } catch {
    /* storage unavailable */
  }
  return "all";
};

// =============================================================================
// Types
// =============================================================================

// =============================================================================
// Utility Functions
// =============================================================================

const shuffleArray = <T,>(array: readonly T[]): T[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};

const createInitialProgress = (): Map<string, WordProgress> =>
  new Map(ALL_WORDS.map(({ english }) => [english, newWordProgress(english)]));

const loadProgressFromStorage = (): Map<string, WordProgress> => {
  const saved = localStorage.getItem("wordProgress");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      // Repair values saved by the old (buggy) ease-factor update.
      return new Map(
        Object.entries(parsed as Record<string, WordProgress>).map(([w, p]) => [w, sanitizeProgress(p)] as const),
      );
    } catch {
      // If parsing fails, initialize fresh
    }
  }
  return createInitialProgress();
};

// =============================================================================
// Sub-components
// =============================================================================

interface ProgressBarProps {
  current: number;
  total: number;
}

const ProgressBar = ({ current, total }: ProgressBarProps) => {
  const percentage = Math.round((current / total) * 100);

  return (
    <div
      className="w-full bg-sage-50 rounded-full h-2 overflow-hidden"
      role="progressbar"
      aria-valuenow={current}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-label={`Progress: ${current} of ${total} words`}
    >
      <div
        className="bg-sage-600 h-full transition-all duration-300 ease-out"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
};

interface ProgressStatsProps {
  current: number;
  total: number;
  practiced: number;
  mastered: number;
}

const ProgressStats = ({
  current,
  total,
  practiced,
  mastered,
}: ProgressStatsProps) => (
  <div className="flex items-center justify-center gap-6 mb-1 text-sm">
    <span className="font-semibold text-ink">
      {current} of {total}
    </span>
    <span className="text-ink-muted hidden sm:inline">
      Practiced: {practiced}
    </span>
    <span className="text-ink-muted hidden sm:inline">
      Mastered: {mastered}
    </span>
  </div>
);

interface WordBadgesProps {
  tricky?: boolean;
  accuracy?: number;
  onCompare?: () => void;
}

/** "tricky sign" tag and "Compare with…" button, shown inline with the word. */
const WordBadges = ({ tricky, accuracy, onCompare }: WordBadgesProps) =>
  tricky || onCompare ? (
    <div className="flex flex-wrap items-center gap-2 self-center">
      {tricky && (
        <HeaderChip
          title={
            accuracy !== undefined
              ? `Fluent signers were recognised ${Math.round(accuracy * 100)}% of the time on this sign.`
              : undefined
          }
        >
          <Flame className="h-3.5 w-3.5" aria-hidden="true" />
          tricky sign
        </HeaderChip>
      )}
      {onCompare && (
        <PillButton size="sm" variant="secondary" icon={<GitCompare className="h-4 w-4" />} onClick={onCompare}>
          Compare with…
        </PillButton>
      )}
    </div>
  ) : null;

interface WordDisplayProps {
  english: string;
  korean: string;
  /** Tags shown inline after the word ("tricky sign", "Compare with…"). */
  badges?: ReactNode;
  onPrevious: () => void;
  onNext: () => void;
  canGoPrevious: boolean;
}

const WordDisplay = ({
  english,
  korean,
  badges,
  onPrevious,
  onNext,
  canGoPrevious,
}: WordDisplayProps) => (
  <div className="flex items-center justify-between gap-4">
    <button
      type="button"
      onClick={onPrevious}
      disabled={!canGoPrevious}
      className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-full text-ink-muted hover:bg-sage-50 disabled:opacity-30 disabled:pointer-events-none transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600"
      aria-label="Previous word"
    >
      <ChevronLeft className="w-7 h-7" aria-hidden="true" />
    </button>

    <div className="flex flex-col items-center flex-1 space-y-1">
      <p className="sr-only">Sign this word</p>
      <div className="flex flex-wrap items-baseline justify-center gap-x-4">
        <h2 className="text-4xl md:text-5xl font-semibold tracking-tight text-ink">
          {english}
        </h2>
        <p className="text-2xl md:text-3xl font-semibold text-primary" lang="ko">
          {korean}
        </p>
        {badges}
      </div>
    </div>

    <button
      type="button"
      onClick={onNext}
      className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-full text-ink-muted hover:bg-sage-50 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600"
      aria-label="Next word"
    >
      <ChevronRight className="w-7 h-7" aria-hidden="true" />
    </button>
  </div>
);

interface ResetDialogProps {
  onReset: () => void;
}

const ResetDialog = ({ onReset }: ResetDialogProps) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <PillButton variant="ghost" onClick={() => setOpen(true)} icon={<RotateCcw className="h-4 w-4" />} className="px-3 md:px-5 text-danger hover:bg-sage-50">
        <span className="hidden md:inline">Reset</span>
        <span className="sr-only md:hidden">Reset</span>
      </PillButton>
      <AppDialog
        alert
        open={open}
        onOpenChange={setOpen}
        title="Reset all progress?"
        description="This will clear all your learning progress and cannot be undone. You will start fresh with all words marked as unlearned."
        footer={
          <PillButtonRow>
            <PillButton variant="secondary" onClick={() => setOpen(false)}>
              Cancel
            </PillButton>
            <PillButton
              variant="danger"
              onClick={() => {
                onReset();
                setOpen(false);
              }}
            >
              Reset Progress
            </PillButton>
          </PillButtonRow>
        }
      />
    </>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const Learn = () => {
  const navigate = useNavigate();
  const [isLoaded, setIsLoaded] = useState(false);

  const [wordProgress, setWordProgress] = useState<Map<string, WordProgress>>(
    loadProgressFromStorage,
  );

  const [deck, setDeck] = useState<DeckChoice>(loadDeckChoice);
  const decks = useMemo(() => availableDecks(), []);
  const deckWords = useMemo(() => wordsInDeck(deck).map((w) => w.english), [deck]);
  const [compareOpen, setCompareOpen] = useState(false);

  const [practiceQueue, setPracticeQueue] = useState<string[]>(() =>
    shuffleArray(wordsInDeck(loadDeckChoice()).map((w) => w.english)),
  );

  const [currentIndex, setCurrentIndex] = useState(0);

  // Persist progress to localStorage
  useEffect(() => {
    const progressObj = Object.fromEntries(wordProgress);
    localStorage.setItem("wordProgress", JSON.stringify(progressObj));
  }, [wordProgress]);

  // Initial load animation
  useEffect(() => {
    const timeout = setTimeout(() => setIsLoaded(true), 50);
    return () => clearTimeout(timeout);
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          handlePrevious();
          break;
        case "ArrowRight":
          e.preventDefault();
          handleNext();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentIndex, practiceQueue.length]);

  // Derived state
  const currentWord = practiceQueue[currentIndex];
  const currentWordData = wordEntry(currentWord);
  const totalWords = deckWords.length;
  const compareWith = currentWord ? compareCandidates(currentWord) : [];

  // Progress stats are scoped to the selected deck (spaced-repetition state itself is per word).
  const deckProgress = deckWords
    .map((w) => wordProgress.get(w))
    .filter((p): p is WordProgress => p !== undefined);
  const practicedWords = deckProgress.filter(
    (p) => p.correctCount > 0 || p.incorrectCount > 0,
  ).length;

  const masteredWords = deckProgress.filter(
    (p) =>
      p.correctCount >= MASTERY_CORRECT_THRESHOLD &&
      p.interval >= MASTERY_INTERVAL_DAYS,
  ).length;

  // Navigation handlers
  const handleNext = useCallback(() => {
    if (currentIndex < practiceQueue.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    } else {
      setPracticeQueue(shuffleArray(deckWords));
      setCurrentIndex(0);
    }
  }, [currentIndex, practiceQueue.length, deckWords]);

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  }, [currentIndex]);

  const handleShuffle = useCallback(() => {
    setPracticeQueue(shuffleArray(deckWords));
    setCurrentIndex(0);
  }, [deckWords]);

  const handleDeckChange = useCallback((value: string) => {
    if (!isDeckChoice(value)) return;
    setDeck(value);
    try {
      localStorage.setItem(DECK_STORAGE_KEY, value);
    } catch {
      /* preference just isn't remembered */
    }
    setPracticeQueue(shuffleArray(wordsInDeck(value).map((w) => w.english)));
    setCurrentIndex(0);
  }, []);

  const handleReset = useCallback(() => {
    setWordProgress(createInitialProgress());
    localStorage.removeItem("wordProgress");
    handleShuffle();
  }, [handleShuffle]);

  const updateProgress = useCallback((word: string, correct: boolean) => {
    setWordProgress((prev) => {
      const newProgress = new Map(prev);
      const current = newProgress.get(word) || newWordProgress(word);

      newProgress.set(word, calculateNextReview(current, correct));
      return newProgress;
    });
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-background [border-bottom:var(--card-border)]">
        <div className="max-w-7xl mx-auto px-6 py-2">
          {/* Top row: Logo and action buttons */}
          <div className="flex items-center justify-between mb-1">
            <img
              src="/koala_logo.png"
              alt="Koala - Go to homepage"
              className="h-10 md:h-11 w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen mt-2 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
              onClick={() => navigate("/")}
              onKeyDown={(e) => e.key === "Enter" && navigate("/")}
              tabIndex={0}
              role="button"
            />

            <div className="flex items-center gap-1 sm:gap-2">
              <HeaderSelect
                aria-label="Choose a deck"
                value={deck}
                onValueChange={handleDeckChange}
                className="w-[132px] sm:w-[190px] md:w-[240px]"
                options={[
                  { value: "all", label: `All words (${AVAILABLE_WORDS.length})` },
                  ...decks.map((d) => ({ value: d.id, label: `${d.label} (${d.count})` })),
                ]}
              />
              <PillButton variant="ghost" onClick={handleShuffle} icon={<Shuffle className="h-4 w-4" />} className="px-3 md:px-5">
                <span className="hidden md:inline">Shuffle</span>
                <span className="sr-only md:hidden">Shuffle</span>
              </PillButton>
              <ResetDialog onReset={handleReset} />
            </div>
          </div>

          {/* Progress stats */}
          <ProgressStats
            current={currentIndex + 1}
            total={totalWords}
            practiced={practicedWords}
            mastered={masteredWords}
          />

          {/* Progress bar */}
          <ProgressBar current={currentIndex + 1} total={totalWords} />
        </div>
      </header>

      {/* Live region for screen reader announcements */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {currentWordData &&
          `Current word: ${currentWordData.english}, Korean: ${currentWordData.korean}`}
      </div>

      {/* Main content: word header, then the Learn -> Practice stage */}
      <main
        className={`
          flex-1 flex flex-col items-stretch
          max-w-6xl mx-auto w-full px-4 sm:px-6 gap-4 pt-[120px] pb-10
          transition-opacity duration-700 ease-out
          ${isLoaded ? "opacity-100" : "opacity-0"}
        `}
      >
        {currentWordData && (
          <WordDisplay
            english={currentWordData.english}
            korean={currentWordData.korean}
            badges={
              <WordBadges
                tricky={isTricky(currentWordData.english)}
                accuracy={heldOutAccuracy(currentWordData.english)}
                onCompare={compareWith.length > 0 ? () => setCompareOpen(true) : undefined}
              />
            }
            onPrevious={handlePrevious}
            onNext={handleNext}
            canGoPrevious={currentIndex > 0}
          />
        )}

        {currentWord && (
          <Stage
            word={currentWord}
            onNext={handleNext}
            onFeedback={updateProgress}
          />
        )}

        <p className="text-xs text-ink-muted text-center">
          Clips trimmed from the original source videos:{" "}
          <a
            href="https://doi.org/10.1007/978-3-030-37731-1_43"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-ink transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600"
          >
            Yang et al., "The Korean Sign Language Dataset for Action Recognition," MMM 2020
          </a>
        </p>
      </main>

      {currentWord && compareWith.length > 0 && (
        <CompareDialog
          word={currentWord}
          others={compareWith}
          open={compareOpen}
          onOpenChange={setCompareOpen}
        />
      )}
    </div>
  );
};

export default Learn;
