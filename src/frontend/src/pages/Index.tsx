import { Hero } from "@/components/Hero";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  const handleStartLearning = () => {
    navigate("/learn");
  };

  return <Hero onStartLearning={handleStartLearning} />;
};

export default Index;
