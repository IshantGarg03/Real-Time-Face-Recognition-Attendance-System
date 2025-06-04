import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Camera } from "lucide-react";
import { motion } from "framer-motion";

export default function FaceAttendancePage() {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("Click to mark attendance");

  const handleAttendance = async () => {
    setLoading(true);
    setMessage("Recognizing your face...");

    try {
      const res = await fetch("http://localhost:5000/mark-attendance", {
        method: "POST",
      });
      const data = await res.json();
      setMessage(`${data.name} marked as ${data.action} at ${data.timestamp}`);
    } catch (err) {
      setMessage("Error marking attendance. Please try again.");
    }

    setLoading(false);
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900 p-4">
      <Card className="w-full max-w-lg rounded-2xl shadow-lg text-white bg-gray-950 border-gray-700">
        <CardContent className="p-8">
          <h1 className="text-3xl font-bold mb-4 text-center">
            Face Attendance System
          </h1>
          <p className="text-center text-gray-400 mb-6">Using facial recognition and location logging</p>

          <motion.div
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <Button
              onClick={handleAttendance}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white w-full py-2 px-4 text-lg rounded-xl"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Camera className="mr-2 h-5 w-5" />
                  Start Camera & Mark Attendance
                </>
              )}
            </Button>
            <p className="mt-6 text-center text-sm text-gray-300">{message}</p>
          </motion.div>
        </CardContent>
      </Card>
    </main>
  );
}





// console.log("Node.js is running inside VS Code");