import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { LearningCard } from "@/components/LearningCard";
import { VideoExampleCard } from "@/components/VideoExampleCard";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { RotateCcw, Shuffle, ChevronLeft, ChevronRight } from "lucide-react";

// =============================================================================
// Constants
// =============================================================================

const MASTERY_CORRECT_THRESHOLD = 3;
const MASTERY_INTERVAL_DAYS = 7;
const INITIAL_EASE_FACTOR = 2.5;
const MIN_EASE_FACTOR = 1.3;
const EASE_FACTOR_DECREMENT = 0.2;

// Word list with Korean translations
const WORD_LIST = [
  { english: "hi", korean: "안녕" },
  { english: "meet", korean: "만나다" },
  { english: "glad", korean: "기쁘다" },
  { english: "me", korean: "나" },
  { english: "name", korean: "이름" },
  { english: "equal", korean: "같다" },
  { english: "eat", korean: "먹다" },
  { english: "do effort", korean: "노력하다" },
  { english: "age", korean: "나이" },
  { english: "again", korean: "다시" },
  { english: "how many", korean: "얼마나" },
  { english: "day", korean: "날" },
  { english: "when", korean: "언제" },
  { english: "subway", korean: "지하철" },
  { english: "family", korean: "가족" },
  { english: "please", korean: "부탁하다" },
  { english: "sister", korean: "언니/누나" },
  { english: "study", korean: "공부하다" },
  { english: "human", korean: "사람" },
  { english: "now", korean: "지금" },
  { english: "end", korean: "끝" },
  { english: "you", korean: "당신" },
  { english: "worried", korean: "걱정하다" },
  { english: "marry", korean: "결혼하다" },
  { english: "no", korean: "아니요" },
  { english: "sweat", korean: "땀" },
  { english: "yet", korean: "아직" },
  { english: "born", korean: "태어나다" },
  { english: "Seoul", korean: "서울" },
  { english: "dinner", korean: "저녁" },
  { english: "food", korean: "음식" },
] as const;

// Derived constant to avoid repeated mapping
const ENGLISH_WORDS = WORD_LIST.map((w) => w.english);

export const getWords = () => [...ENGLISH_WORDS];

// =============================================================================
// Types
// =============================================================================

interface WordProgress {
  word: string;
  correctCount: number;
  incorrectCount: number;
  lastSeen: number;
  interval: number;
  easeFactor: number;
}

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

const createInitialProgress = (): Map<string, WordProgress> => {
  const progress = new Map<string, WordProgress>();
  WORD_LIST.forEach(({ english }) => {
    progress.set(english, {
      word: english,
      correctCount: 0,
      incorrectCount: 0,
      lastSeen: 0,
      interval: 0,
      easeFactor: INITIAL_EASE_FACTOR,
    });
  });
  return progress;
};

const loadProgressFromStorage = (): Map<string, WordProgress> => {
  const saved = localStorage.getItem("wordProgress");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      return new Map(Object.entries(parsed));
    } catch {
      // If parsing fails, initialize fresh
    }
  }
  return createInitialProgress();
};

/**
 * SM-2 Spaced Repetition Algorithm
 * Updates the ease factor and interval based on response quality
 */
const calculateNextReview = (
  current: WordProgress,
  correct: boolean,
): WordProgress => {
  const updated = { ...current, lastSeen: Date.now() };

  if (correct) {
    updated.correctCount += 1;

    // SM-2: Quality of 4 (correct with hesitation)
    const quality = 4;
    updated.easeFactor = Math.max(
      MIN_EASE_FACTOR,
      current.easeFactor +
        (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    );

    // Calculate next interval
    if (current.interval === 0) {
      updated.interval = 1;
    } else if (current.interval === 1) {
      updated.interval = 6;
    } else {
      updated.interval = Math.round(current.interval * updated.easeFactor);
    }
  } else {
    updated.incorrectCount += 1;
    updated.interval = 0;
    updated.easeFactor = Math.max(
      MIN_EASE_FACTOR,
      current.easeFactor - EASE_FACTOR_DECREMENT,
    );
  }

  return updated;
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
      className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2 overflow-hidden"
      role="progressbar"
      aria-valuenow={current}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-label={`Progress: ${current} of ${total} words`}
    >
      <div
        className="bg-primary h-full transition-all duration-300 ease-out"
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
  <div className="flex items-center justify-center gap-6 mb-3 text-sm">
    <span className="font-semibold text-gray-900 dark:text-gray-100">
      {current} of {total}
    </span>
    <span className="text-gray-600 dark:text-gray-400 hidden sm:inline">
      Practiced: {practiced}
    </span>
    <span className="text-gray-600 dark:text-gray-400 hidden sm:inline">
      Mastered: {mastered}
    </span>
  </div>
);

interface WordDisplayProps {
  english: string;
  korean: string;
  onPrevious: () => void;
  onNext: () => void;
  canGoPrevious: boolean;
}

const WordDisplay = ({
  english,
  korean,
  onPrevious,
  onNext,
  canGoPrevious,
}: WordDisplayProps) => (
  <div className="flex items-center justify-between gap-4">
    <button
      onClick={onPrevious}
      disabled={!canGoPrevious}
      className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
      aria-label="Previous word"
    >
      <ChevronLeft className="w-8 h-8 text-gray-700 dark:text-gray-300" />
    </button>

    <div className="flex flex-col items-center flex-1 space-y-2">
      <p className="text-base md:text-lg font-medium text-gray-600 dark:text-gray-400">
        Sign this word
      </p>
      <h2 className="text-4xl md:text-6xl font-semibold tracking-tight text-gray-900 dark:text-gray-100">
        {english}
      </h2>
      <p className="text-2xl md:text-4xl font-semibold text-primary" lang="ko">
        {korean}
      </p>
    </div>

    <button
      onClick={onNext}
      className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 transition-all"
      aria-label="Next word"
    >
      <ChevronRight className="w-8 h-8 text-gray-700 dark:text-gray-300" />
    </button>
  </div>
);

interface ResetDialogProps {
  onReset: () => void;
}

const ResetDialog = ({ onReset }: ResetDialogProps) => (
  <AlertDialog>
    <AlertDialogTrigger asChild>
      <Button
        size="sm"
        variant="ghost"
        className="rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 px-3 py-2 text-red-600 dark:text-red-400"
      >
        <RotateCcw className="h-4 w-4 md:mr-2" />
        <span className="hidden md:inline">Reset</span>
      </Button>
    </AlertDialogTrigger>
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>Reset all progress?</AlertDialogTitle>
        <AlertDialogDescription>
          This will clear all your learning progress and cannot be undone. You
          will start fresh with all words marked as unlearned.
        </AlertDialogDescription>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogCancel>Cancel</AlertDialogCancel>
        <AlertDialogAction
          onClick={onReset}
          className="bg-red-600 hover:bg-red-700"
        >
          Reset Progress
        </AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  </AlertDialog>
);

// =============================================================================
// Main Component
// =============================================================================

const Learn = () => {
  const navigate = useNavigate();
  const [isLoaded, setIsLoaded] = useState(false);

  const [wordProgress, setWordProgress] = useState<Map<string, WordProgress>>(
    loadProgressFromStorage,
  );

  const [practiceQueue, setPracticeQueue] = useState<string[]>(() =>
    shuffleArray(ENGLISH_WORDS),
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
  const currentWordData = WORD_LIST.find((w) => w.english === currentWord);
  const totalWords = WORD_LIST.length;

  const practicedWords = Array.from(wordProgress.values()).filter(
    (p) => p.correctCount > 0 || p.incorrectCount > 0,
  ).length;

  const masteredWords = Array.from(wordProgress.values()).filter(
    (p) =>
      p.correctCount >= MASTERY_CORRECT_THRESHOLD &&
      p.interval >= MASTERY_INTERVAL_DAYS,
  ).length;

  // Navigation handlers
  const handleNext = useCallback(() => {
    if (currentIndex < practiceQueue.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    } else {
      setPracticeQueue(shuffleArray(ENGLISH_WORDS));
      setCurrentIndex(0);
    }
  }, [currentIndex, practiceQueue.length]);

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  }, [currentIndex]);

  const handleShuffle = useCallback(() => {
    setPracticeQueue(shuffleArray(ENGLISH_WORDS));
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
      const current = newProgress.get(word) || {
        word,
        correctCount: 0,
        incorrectCount: 0,
        lastSeen: 0,
        interval: 0,
        easeFactor: INITIAL_EASE_FACTOR,
      };

      newProgress.set(word, calculateNextReview(current, correct));
      return newProgress;
    });
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col bg-white dark:bg-black">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-black/80 border-b border-gray-200/20 dark:border-gray-800/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          {/* Top row: Logo and action buttons */}
          <div className="flex items-center justify-between mb-4">
            <img
              src="/koala_logo.svg"
              alt="Koala - Go to homepage"
              className="h-14 md:h-14 w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen mt-2 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
              onClick={() => navigate("/")}
              onKeyDown={(e) => e.key === "Enter" && navigate("/")}
              tabIndex={0}
              role="button"
            />

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={handleShuffle}
                className="rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 px-3 py-2"
              >
                <Shuffle className="h-4 w-4 md:mr-2" aria-hidden="true" />
                <span className="hidden md:inline">Shuffle</span>
              </Button>
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

      {/* Main content */}
      <main
        className={`
          flex-1 flex flex-col lg:flex-row items-start justify-center 
          max-w-7xl mx-auto w-full px-6 gap-12 pt-40 pb-12
          transition-opacity duration-700 ease-out
          ${isLoaded ? "opacity-100" : "opacity-0"}
        `}
      >
        {/* Video example section */}
        <div className="flex-1 w-full max-w-2xl mt-[176px]">
          <VideoExampleCard word={currentWord} />
        </div>

        {/* Practice section */}
        <div className="flex-1 w-full max-w-2xl space-y-8">
          {/* Word header with navigation */}
          {currentWordData && (
            <WordDisplay
              english={currentWordData.english}
              korean={currentWordData.korean}
              onPrevious={handlePrevious}
              onNext={handleNext}
              canGoPrevious={currentIndex > 0}
            />
          )}

          {/* Learning Card */}
          <LearningCard
            word={currentWord}
            onNext={handleNext}
            onPrevious={handlePrevious}
            onFeedback={updateProgress}
          />
        </div>
      </main>
    </div>
  );
};

export default Learn;
