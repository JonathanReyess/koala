import { useState, useEffect } from "react";
import { LearningCard } from "@/components/LearningCard";
import { VideoExampleCard } from "@/components/VideoExampleCard";
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { RotateCcw, Shuffle } from "lucide-react";

// Word list with Korean translations
const wordList = [
  { english: "hi", korean: "안녕" },
  { english: "meet", korean: "만나다" },
  { english: "me", korean: "나" },
  { english: "see", korean: "보다" },
  { english: "name", korean: "이름" },
  { english: "thank", korean: "감사하다" },
  { english: "equal", korean: "같다" },
  { english: "sorry", korean: "미안하다" },
  { english: "age", korean: "나이" },
  { english: "day", korean: "날" },
  { english: "please?", korean: "부탁하다" },
  { english: "study", korean: "공부하다" },
  { english: "human", korean: "사람" },
  { english: "now", korean: "지금" },
  { english: "test", korean: "시험" },
  { english: "yet", korean: "아직" },
  { english: "finally", korean: "드디어" },
  { english: "dinner", korean: "저녁" },
  { english: "experience", korean: "경험" },
  { english: "invite", korean: "초대하다" },
  { english: "food", korean: "음식" },
  { english: "want", korean: "원하다" },
  { english: "good", korean: "좋다" },
  { english: "care", korean: "돌보다" }
];

export const getWords = () => wordList.map(w => w.english);

// Spaced repetition tracking
interface WordProgress {
  word: string;
  correctCount: number;
  incorrectCount: number;
  lastSeen: number;
  interval: number; // Days until review
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
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  // Initialize progress tracking from localStorage
  const [wordProgress, setWordProgress] = useState<Map<string, WordProgress>>(() => {
    const saved = localStorage.getItem("wordProgress");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return new Map(Object.entries(parsed));
      } catch {
        // If parsing fails, initialize fresh
      }
    }
    // Initialize all words with default progress
    const initialProgress = new Map<string, WordProgress>();
    wordList.forEach(({ english }) => {
      initialProgress.set(english, {
        word: english,
        correctCount: 0,
        incorrectCount: 0,
        lastSeen: 0,
        interval: 0,
        easeFactor: 2.5
      });
    });
    return initialProgress;
  });

  const [practiceQueue, setPracticeQueue] = useState<string[]>(() => {
    return shuffleArray(wordList.map(w => w.english));
  });

  const [currentIndex, setCurrentIndex] = useState(0);

  // Save progress to localStorage whenever it changes
  useEffect(() => {
    const progressObj = Object.fromEntries(wordProgress);
    localStorage.setItem("wordProgress", JSON.stringify(progressObj));
  }, [wordProgress]);

  const currentWord = practiceQueue[currentIndex];
  const currentWordData = wordList.find(w => w.english === currentWord);
  const progress = wordProgress.get(currentWord);

  // Calculate statistics
  const totalWords = wordList.length;
  const practicedWords = Array.from(wordProgress.values()).filter(
    p => p.correctCount > 0 || p.incorrectCount > 0
  ).length;
  const masteredWords = Array.from(wordProgress.values()).filter(
    p => p.correctCount >= 3 && p.interval >= 7
  ).length;

  const handleNext = () => {
    if (currentIndex < practiceQueue.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else {
      // Completed all words, reshuffle and start over
      const newQueue = shuffleArray(wordList.map(w => w.english));
      setPracticeQueue(newQueue);
      setCurrentIndex(0);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  };

  const handleShuffle = () => {
    const newQueue = shuffleArray(wordList.map(w => w.english));
    setPracticeQueue(newQueue);
    setCurrentIndex(0);
  };

  const handleReset = () => {
    if (confirm("Are you sure you want to reset all progress? This cannot be undone.")) {
      const initialProgress = new Map<string, WordProgress>();
      wordList.forEach(({ english }) => {
        initialProgress.set(english, {
          word: english,
          correctCount: 0,
          incorrectCount: 0,
          lastSeen: 0,
          interval: 0,
          easeFactor: 2.5
        });
      });
      setWordProgress(initialProgress);
      localStorage.removeItem("wordProgress");
      handleShuffle();
    }
  };

  // Update progress based on user performance (SM-2 algorithm)
  const updateProgress = (word: string, correct: boolean) => {
    setWordProgress(prev => {
      const newProgress = new Map(prev);
      const current = newProgress.get(word) || {
        word,
        correctCount: 0,
        incorrectCount: 0,
        lastSeen: 0,
        interval: 0,
        easeFactor: 2.5
      };

      const now = Date.now();
      
      if (correct) {
        current.correctCount += 1;
        // SM-2 spaced repetition algorithm
        const quality = 4; // Good response
        current.easeFactor = Math.max(
          1.3, 
          current.easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
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
        current.interval = 0; // Reset to beginning
        current.easeFactor = Math.max(1.3, current.easeFactor - 0.2);
      }

      current.lastSeen = now;
      newProgress.set(word, current);
      return newProgress;
    });
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-[hsl(var(--background))]">
      {/* Progress Bar - Fixed at top */}
      <div className="fixed top-0 left-0 right-0 bg-[hsl(var(--card))] border-b border-[hsl(var(--border))] py-4 px-6 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-3 gap-4">
            {/* Logo */}
            <img
              src="/koala_logo.svg"
              alt="Koala Logo"
              style={{ width: isMobile ? "120px" : "180px", height: "auto" }}
              className="cursor-pointer hover:opacity-90 transition-opacity flex-shrink-0"
              onClick={() => navigate("/")}
            />

            {/* Progress Stats */}
            <div className="flex items-center gap-4 flex-wrap flex-1 justify-center">
              <span className="text-sm font-semibold text-[hsl(var(--foreground))]">
                Progress: {currentIndex + 1} / {totalWords}
              </span>
              {!isMobile && (
                <span className="text-sm text-[hsl(var(--muted-foreground))]">
                  Practiced: {practicedWords} | Mastered: {masteredWords}
                </span>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 flex-shrink-0">
              <Button
                size="sm"
                variant="outline"
                onClick={handleShuffle}
                className="flex items-center gap-1"
              >
                <Shuffle className="h-4 w-4" />
                {!isMobile && "Shuffle"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleReset}
                className="flex items-center gap-1 text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))]"
              >
                <RotateCcw className="h-4 w-4" />
                {!isMobile && "Reset"}
              </Button>
            </div>
          </div>
          
          {/* Mobile Stats - Show on second line for mobile */}
          {isMobile && (
            <div className="text-center mb-2">
              <span className="text-xs text-[hsl(var(--muted-foreground))]">
                Practiced: {practicedWords} | Mastered: {masteredWords}
              </span>
            </div>
          )}

          {/* Progress Bar */}
          <div className="w-full bg-[hsl(var(--muted))] rounded-full h-3.5 overflow-hidden">
            <div
              className="bg-[hsl(var(--accent))] h-full transition-all duration-300 ease-in-out"
              style={{ width: `${((currentIndex + 1) / totalWords) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="flex-1 flex flex-col md:flex-row items-start justify-center px-4 gap-8 pt-32 md:pt-40 pb-8">
        {/* Example video */}
        <div className="flex-1 w-full max-w-xl flex flex-col justify-start mt-40">
          <VideoExampleCard word={currentWord} />
        </div>

        {/* Learning card + header */}
        <div className="flex-1 w-full max-w-xl flex flex-col justify-start mt-3.5">
          {/* Header with word and arrows */}
          <div className="flex items-center justify-between w-full mb-6 gap-4">
            {/* Previous Button */}
            <button
              onClick={handlePrevious}
              disabled={currentIndex === 0}
              className="p-2 hover:opacity-80 disabled:opacity-30 disabled:cursor-not-allowed transition-opacity flex-shrink-0"
              aria-label="Previous word"
            >
              <svg
                fill="#b6d3b7"
                viewBox="0 0 32 32"
                className="w-8 h-8 md:w-10 md:h-10"
                style={{ transform: "scaleX(-1)" }}
              >
                <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
              </svg>
            </button>

            {/* Word header with Korean translation */}
            <div className="flex flex-col items-center flex-1">
              <p className="text-base md:text-xl font-semibold mb-1 tracking-tight text-[#878787]">
                Sign the following word:
              </p>
              <h2 className="text-3xl md:text-5xl font-extrabold text-[hsl(var(--primary))]">
                {currentWordData?.english}
              </h2>
              <p className="text-xl md:text-3xl font-bold text-[hsl(var(--accent))] mt-1">
                {currentWordData?.korean}
              </p>
              {progress && (progress.correctCount > 0 || progress.incorrectCount > 0) && (
                <div className="flex gap-3 mt-2 text-sm">
                  <span className="text-green-600 font-medium">✓ {progress.correctCount}</span>
                  <span className="text-red-600 font-medium">✗ {progress.incorrectCount}</span>
                </div>
              )}
            </div>

            {/* Next Button */}
            <button
              onClick={handleNext}
              className="p-2 hover:opacity-80 transition-opacity flex-shrink-0"
              aria-label="Next word"
            >
              <svg
                fill="#b6d3b7"
                viewBox="0 0 32 32"
                className="w-8 h-8 md:w-10 md:h-10"
              >
                <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
              </svg>
            </button>
          </div>

          {/* Learning Card - Pass updateProgress callback */}
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