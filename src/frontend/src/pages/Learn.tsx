import { useState, useEffect } from "react";
import { LearningCard } from "@/components/LearningCard";
import { VideoExampleCard } from "@/components/VideoExampleCard";
import { Button } from "@/components/ui/button";
import { RotateCcw, Shuffle, ChevronLeft, ChevronRight } from "lucide-react";

// Word list with Korean translations
const wordList = [
  { english: "hi", korean: "안녕" },
  { english: "meet", korean: "고기" },
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
];

export const getWords = () => wordList.map((w) => w.english);

interface WordProgress {
  word: string;
  correctCount: number;
  incorrectCount: number;
  lastSeen: number;
  interval: number;
  easeFactor: number;
}

const shuffleArray = <T,>(array: T[]): T[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};

const Learn = () => {
  const [isLoaded, setIsLoaded] = useState(false);

  const [wordProgress, setWordProgress] = useState<Map<string, WordProgress>>(
    () => {
      const saved = localStorage.getItem("wordProgress");
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          return new Map(Object.entries(parsed));
        } catch {
          // If parsing fails, initialize fresh
        }
      }
      const initialProgress = new Map<string, WordProgress>();
      wordList.forEach(({ english }) => {
        initialProgress.set(english, {
          word: english,
          correctCount: 0,
          incorrectCount: 0,
          lastSeen: 0,
          interval: 0,
          easeFactor: 2.5,
        });
      });
      return initialProgress;
    }
  );

  const [practiceQueue, setPracticeQueue] = useState<string[]>(() => {
    return shuffleArray(wordList.map((w) => w.english));
  });

  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const progressObj = Object.fromEntries(wordProgress);
    localStorage.setItem("wordProgress", JSON.stringify(progressObj));
  }, [wordProgress]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setIsLoaded(true);
    }, 50);
    return () => clearTimeout(timeout);
  }, []);

  const currentWord = practiceQueue[currentIndex];
  const currentWordData = wordList.find((w) => w.english === currentWord);

  const totalWords = wordList.length;
  const practicedWords = Array.from(wordProgress.values()).filter(
    (p) => p.correctCount > 0 || p.incorrectCount > 0
  ).length;
  const masteredWords = Array.from(wordProgress.values()).filter(
    (p) => p.correctCount >= 3 && p.interval >= 7
  ).length;

  const handleNext = () => {
    if (currentIndex < practiceQueue.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    } else {
      const newQueue = shuffleArray(wordList.map((w) => w.english));
      setPracticeQueue(newQueue);
      setCurrentIndex(0);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  };

  const handleShuffle = () => {
    const newQueue = shuffleArray(wordList.map((w) => w.english));
    setPracticeQueue(newQueue);
    setCurrentIndex(0);
  };

  const handleReset = () => {
    if (
      confirm(
        "Are you sure you want to reset all progress? This cannot be undone."
      )
    ) {
      const initialProgress = new Map<string, WordProgress>();
      wordList.forEach(({ english }) => {
        initialProgress.set(english, {
          word: english,
          correctCount: 0,
          incorrectCount: 0,
          lastSeen: 0,
          interval: 0,
          easeFactor: 2.5,
        });
      });
      setWordProgress(initialProgress);
      localStorage.removeItem("wordProgress");
      handleShuffle();
    }
  };

  const updateProgress = (word: string, correct: boolean) => {
    setWordProgress((prev) => {
      const newProgress = new Map(prev);
      const current = newProgress.get(word) || {
        word,
        correctCount: 0,
        incorrectCount: 0,
        lastSeen: 0,
        interval: 0,
        easeFactor: 2.5,
      };

      const now = Date.now();

      if (correct) {
        current.correctCount += 1;
        const quality = 4;
        current.easeFactor = Math.max(
          1.3,
          current.easeFactor +
            (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        );

        if (current.interval === 0) {
          current.interval = 1;
        } else if (current.interval === 1) {
          current.interval = 6;
        } else {
          current.interval = Math.round(current.interval * current.easeFactor);
        }
      } else {
        current.incorrectCount += 1;
        current.interval = 0;
        current.easeFactor = Math.max(1.3, current.easeFactor - 0.2);
      }

      current.lastSeen = now;
      newProgress.set(word, current);
      return newProgress;
    });
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-white dark:bg-black">
      {/* Header with backdrop blur - Apple style */}
      <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-black/80 border-b border-gray-200/20 dark:border-gray-800/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          {/* Top row: Logo and action buttons */}
          <div className="flex items-center justify-between mb-4">
            <img
              src="/koala_logo.svg"
              alt="Koala Logo"
              className="h-14 md:h-14 w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen mt-2"
              onClick={() => (window.location.href = "/")}
            />

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={handleShuffle}
                className="rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 px-3 py-2"
              >
                <Shuffle className="h-4 w-4 md:mr-2" />
                <span className="hidden md:inline">Shuffle</span>
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={handleReset}
                className="rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 px-3 py-2 text-red-600 dark:text-red-400"
              >
                <RotateCcw className="h-4 w-4 md:mr-2" />
                <span className="hidden md:inline">Reset</span>
              </Button>
            </div>
          </div>

          {/* Progress stats */}
          <div className="flex items-center justify-center gap-6 mb-3 text-sm">
            <span className="font-semibold text-gray-900 dark:text-gray-100">
              {currentIndex + 1} of {totalWords}
            </span>
            <span className="text-gray-600 dark:text-gray-400 hidden sm:inline">
              Practiced: {practicedWords}
            </span>
            <span className="text-gray-600 dark:text-gray-400 hidden sm:inline">
              Mastered: {masteredWords}
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-300 ease-out"
              style={{ width: `${((currentIndex + 1) / totalWords) * 100}%` }}
            />
          </div>
        </div>
      </header>

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
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={handlePrevious}
              disabled={currentIndex === 0}
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
                {currentWordData?.english}
              </h2>
              <p className="text-2xl md:text-4xl font-semibold text-primary">
                {currentWordData?.korean}
              </p>
            </div>

            <button
              onClick={handleNext}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-900 transition-all"
              aria-label="Next word"
            >
              <ChevronRight className="w-8 h-8 text-gray-700 dark:text-gray-300" />
            </button>
          </div>

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
