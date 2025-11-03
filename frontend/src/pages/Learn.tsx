import { useState } from "react";
import { LearningCard } from "@/components/LearningCard";
import { VideoExampleCard } from "@/components/VideoExampleCard";
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigate } from "react-router-dom";

// Replace the array export with a getter function
export const getWords = () => [
  "hi", "meet", "me", "see", "name", "thank", "equal", "sorry",
  "age", "how many", "day", "please?",
  "study", "human", "now", "test", "yet", "finally",
  "dinner", "experience", "invite", "food", "want", "good", "care"
];

const Learn = () => {
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  const words = getWords();

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

      {/* Main content */}
      <main className="flex-1 flex flex-col md:flex-row items-center justify-center px-4 gap-8 pt-24">
        {/* Example video */}
        <div className="flex-1 max-w-xl flex flex-col h-full justify-center">
          <VideoExampleCard word={words[currentWordIndex]} />
        </div>

        {/* Learning card + header */}
        <div className="flex-1 max-w-xl flex flex-col h-full justify-center">
          {/* Header with word and arrows */}
          <div className="flex items-center justify-between w-full max-w-2xl mx-auto mt-4 md:-mt-24 mb-6">
            {/* Previous Button */}
            <button
              onClick={handlePrevious}
              className="p-2 hover:opacity-80"
            >
              <svg
                fill="#b6d3b7"
                viewBox="0 0 32 32"
                className="w-10 h-10"
                style={{ transform: "scaleX(-1)" }}
              >
                <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
              </svg>
            </button>

            {/* Word header */}
            <div className="flex flex-col items-center">
              <p className="text-xl md:text-2xl font-semibold mb-2 tracking-tight text-[#878787]">
                Sign the following word:
              </p>
              <h2 className="text-5xl font-extrabold text-[hsl(var(--primary))]">
                {words[currentWordIndex]}
              </h2>
            </div>

            {/* Next Button */}
            <button
              onClick={handleNext}
              className="p-2 hover:opacity-80"
            >
              <svg
                fill="#b6d3b7"
                viewBox="0 0 32 32"
                className="w-10 h-10"
              >
                <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
              </svg>
            </button>
          </div>

          {/* Learning Card */}
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
