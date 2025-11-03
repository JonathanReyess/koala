import { useState } from "react";
import { LearningCard } from "@/components/LearningCard";
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigate } from "react-router-dom";

// Words list for iteration in Learn.tsx
export const words = [
  "hi", "meat", "me", "see", "name", "thank", "equal", "sorry", 
  "age", "how many", "day", "good, nice", "number", "please?", 
  "study", "human", "now", "education", "test", "yet", "finally", 
  "dinner", "experience", "invite", "food", "want", "good", "care"
];

const Learn = () => {
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  // Start at a random index
  const [currentWordIndex, setCurrentWordIndex] = useState(
    Math.floor(Math.random() * words.length)
  );

  const handleNext = () =>
    setCurrentWordIndex((prev) => (prev + 1) % words.length);

  const handlePrevious = () =>
    setCurrentWordIndex((prev) => (prev - 1 + words.length) % words.length);

  return (
    <div className="relative min-h-screen flex flex-col bg-[hsl(var(--background))]">
      {/* Top-left logo */}
      <header className="absolute top-0 left-0 p-6 z-50">
        <img
          src="/koala.svg"
          alt="Koala Logo"
          style={{ width: isMobile ? "172px" : "250px", height: "auto" }}
          className="cursor-pointer hover:opacity-90 transition-opacity"
          onClick={() => navigate("/")}
        />
      </header>

      {/* Centered LearningCard */}
      <main className="flex-1 flex items-center justify-center px-4">
        <div className="w-full max-w-2xl">
          <LearningCard
            word={words[currentWordIndex]}
            onNext={handleNext}
            onPrevious={handlePrevious}
          />
        </div>
      </main>
    </div>
  );
};

export default Learn;
