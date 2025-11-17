import React from 'react';
import { Button } from "@/components/ui/button"; // Assuming this is your Button component
import { Input } from "@/components/ui/input";   // You may need to create or install an Input component
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"; // Assuming you use Card components (like shadcn/ui)
import { Link } from 'react-router-dom'; // Assuming you use react-router-dom for navigation

export const LoginPage: React.FC = () => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement actual log-in logic (API call, state management, etc.)
    console.log('Login attempt...');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <img
            src="/koala_logo.svg"
            alt="Koala Logo"
            className="w-24 h-auto mx-auto mb-4"
          />
          <CardTitle className="text-3xl font-bold">Welcome Back!</CardTitle>
          <CardDescription>
            Sign in to continue mastering Korean Sign Language.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
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
              Log In
            </Button>
          </form>
          <div className="mt-4 text-center text-sm">
            Don't have an account?{' '}
            {/* Make sure your router is set up for the /signup path */}
            <Link to="/signup" className="text-primary hover:underline font-medium">
              Sign Up
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};