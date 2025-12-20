import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Link } from "react-router-dom";

export const SignupPage: React.FC = () => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement actual sign-up/registration logic
    console.log("Signup attempt...");
  };

  return (
    <div
      // 1. Use custom background and add 3D perspective style
      className="min-h-screen flex items-center justify-center p-4 bg-gray-50"
      style={{ perspective: "1000px" }}
    >
      {/* 2. Apply flip animation and reverse starting direction.
          The group-[.animate-flip-in] class will apply the 1s animation,
          and we override the initial transform using an inline class. 
          To make it flip from the right, the initial rotation must be +90deg.
          We use an arbitrary value class `[--flip-start:90deg]` to modify the starting point.
          NOTE: This requires adding a variable to your tailwind keyframe definition (see next section).
      */}
      <Card
        className="w-full max-w-md animate-flip-in"
        style={
          {
            // Override the initial rotation start point for the right-to-left flip
            "--tw-rotate-y-start": "90deg",
          } as React.CSSProperties
        } // Cast required for custom CSS properties in TSX
      >
        <CardHeader className="text-center">
          {/* 3. Wrap logo with Link to home page */}
          <Link
            to="/"
            className="inline-block hover:opacity-80 transition-opacity"
          >
            <img
              src="/koala_logo.svg"
              alt="Go to Home"
              className="w-32 h-auto mx-auto cursor-pointer"
            />
          </Link>
          <CardTitle className="text-3xl font-bold">
            Create Your Account
          </CardTitle>
          <CardDescription>
            Join now and start your journey to master Korean Sign Language.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="name">Name / Username</label>
              <Input
                id="name"
                type="text"
                placeholder="Jonathan Reyes"
                required
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="email">Email</label>
              <Input
                id="email"
                type="email"
                placeholder="your.email@example.com"
                required
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="password">Password</label>
              <Input
                id="password"
                type="password"
                required
                className="w-full"
              />
            </div>
            <Button type="submit" className="w-full py-6 text-lg">
              Sign Up
            </Button>
          </form>
          <div className="mt-4 text-center text-sm">
            Already have an account?{" "}
            <Link
              to="/login"
              className="text-primary hover:underline font-medium"
            >
              Log In
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
