import React from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from 'react-router-dom';

export const SignupPage: React.FC = () => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement actual sign-up/registration logic
    console.log('Signup attempt...');
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
          <CardTitle className="text-3xl font-bold">Create Your Account</CardTitle>
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
            Already have an account?{' '}
            {/* Make sure your router is set up for the /login path */}
            <Link to="/login" className="text-primary hover:underline font-medium">
              Log In
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};