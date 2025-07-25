import { useRef, useState } from "react";
import { transcribeRecording as transcribeRecordingApi, analyseKeyword, analyseContent, analyseSentiment, analyseSentimentModel, summariseFeedback, type AnalysisResponse, type SentimentModelResponse } from "../api/ApiService";
import type { Question } from "../data/questions";
import type { KeywordAnalysis, ResponseContentAnalysis, ResponseSentimentAnalysis, FeedbackSummary } from "../types/feedbackSummariser";

type AnalysisStep = "audio" | "keyword" | "content" | "sentiment_model" | "sentiment" | "summary";
type StepStatus = "pending" | "in_progress" | "done" | "error";
type AnalysisProgress = Record<AnalysisStep, StepStatus>;

function unwrapResult<T>(res: T | { result: T }): T {
  if (res && typeof res === "object" && "result" in res) {
    return (res as { result: T }).result;
  }
  return res as T;
}

export const useRecorder = (currentQuestion: Question | null) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcription, setTranscription] = useState<string>("");
  const [showTranscription, setShowTranscription] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    audio: "pending",
    keyword: "pending",
    content: "pending",
    sentiment_model: "pending",
    sentiment: "pending",
    summary: "pending",
  });

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const saveRecording = () => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "recording.webm";
    a.click();
    URL.revokeObjectURL(url);
  };

  const transcribeRecording = async () => {
    if (recordedChunks.length === 0) return;

    try {
      setIsTranscribing(true);
      setError(null);

      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const transcriptionText = await transcribeRecordingApi(blob);

      setTranscription(transcriptionText);
      setShowTranscription(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setIsTranscribing(false);
    }
  };

  const processAnalysis = async () => {
    if (recordedChunks.length === 0 || !transcription || !currentQuestion) return;

    try {
      setIsProcessing(true);
      setError(null);
      setAnalysisProgress({
        audio: "in_progress",
        keyword: "pending",
        content: "pending",
        sentiment_model: "pending",
        sentiment: "pending",
        summary: "pending",
      });

      let keywordResults: KeywordAnalysis | null = null;
      let contentResults: ResponseContentAnalysis | null = null;
      let sentimentResults: ResponseSentimentAnalysis | null = null;
      let sentimentModelResults: SentimentModelResponse | null = null;
      let feedbackSummary: FeedbackSummary | null = null;
      let audioResults: unknown = null;
      // Audio analysis (if available)
      try {
        setAnalysisProgress((prev) => ({ ...prev, audio: "in_progress" }));
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        audioResults = await import("../api/ApiService").then(api => api.analyseAudio(blob, transcription));
        setAnalysisProgress((prev) => ({ ...prev, audio: "done", keyword: prev.keyword }));
      } catch {
        setAnalysisProgress((prev) => ({ ...prev, audio: "error" }));
      }

      // Keyword analysis
      try {
        setAnalysisProgress((prev) => ({ ...prev, keyword: "in_progress" }));
        const keywordRes = await analyseKeyword(currentQuestion.text, transcription);
        keywordResults = unwrapResult(keywordRes);
        setAnalysisProgress((prev) => ({ ...prev, keyword: "done", content: "in_progress" }));
      } catch (err) {
        setAnalysisProgress((prev) => ({ ...prev, keyword: "error" }));
        throw err;
      }

      // Content analysis
      try {
        const contentRes = await analyseContent(currentQuestion.text, transcription);
        contentResults = unwrapResult(contentRes);
        setAnalysisProgress((prev) => ({ ...prev, content: "done", sentiment_model: "in_progress" }));
      } catch (err) {
        setAnalysisProgress((prev) => ({ ...prev, content: "error" }));
        throw err;
      }

      // Sentiment analysis (model call)
      try {
        setAnalysisProgress((prev) => ({ ...prev, sentiment_model: "in_progress" }));
        sentimentModelResults = await analyseSentimentModel(currentQuestion.text, transcription);
        setAnalysisProgress((prev) => ({ ...prev, sentiment_model: "done", sentiment: "in_progress" }));
      } catch (err) {
        setAnalysisProgress((prev) => ({ ...prev, sentiment_model: "error" }));
        throw err;
      }

      // Sentiment analysis (agent)
      try {
        const sentimentRes = await analyseSentiment(currentQuestion.text, transcription);
        sentimentResults = unwrapResult(sentimentRes);
        setAnalysisProgress((prev) => ({ ...prev, sentiment: "done", summary: "in_progress" }));
      } catch (err) {
        setAnalysisProgress((prev) => ({ ...prev, sentiment: "error" }));
        throw err;
      }

      // Feedback summary
      try {
        feedbackSummary = await summariseFeedback(keywordResults!, contentResults!, sentimentResults!);
        setAnalysisProgress((prev) => ({ ...prev, summary: "done" }));
      } catch (err) {
        setAnalysisProgress((prev) => ({ ...prev, summary: "error" }));
        throw err;
      }

      setAnalysisResults({
        feedbackSummary: feedbackSummary!,
        agentResults: {
          keywordAnalysis: keywordResults!,
          responseContent: contentResults!,
          responseSentiment: sentimentResults!,
        },
        results:
          audioResults && typeof audioResults === "object" && audioResults !== null && Object.keys(audioResults as object).length > 0
            ? (audioResults as { [feature: string]: { Score: number; Feedback: string } })
            : undefined,
        sentiment: sentimentResults?.sentiment || undefined,
        sentimentModelResponse: sentimentModelResults || undefined,
        transcription,
        error: undefined,
      });
      setShowTranscription(false);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      setError(errorMsg);
      setAnalysisResults({
        feedbackSummary: {
          verdict: "",
          strengths: "",
          weaknesses: "",
          improvement_suggestion: "",
          overall_score: 0,
        },
        agentResults: {
          keywordAnalysis: {
            matched_keywords: [],
            missing_keywords: [],
            score: 0,
            notes: "",
          },
          responseContent: {
            assessment: [],
            scores: {
              clarityAndStructure: 0,
              relevance: 0,
              useOfStarMethod: 0,
              impact: 0,
              authenticity: 0,
            },
            improvement: [],
          },
          responseSentiment: {
            sentiment: "",
            confidence: 0,
            evidence: [],
          },
        },
        error: errorMsg,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTranscriptionEdit = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTranscription(event.target.value);
  };

  const handleTranscriptionSubmit = () => {
    if (transcription.trim()) {
      processAnalysis();
    }
  };

  const handleTranscriptionCancel = () => {
    setShowTranscription(false);
    setTranscription("");
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      setStream(stream);
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data]);
        }
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      // Reset previous results when starting a new recording
      setAnalysisResults(null);
      setError(null);
      setTranscription("");
      setShowTranscription(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to access camera/microphone");
    }
  };

  return {
    // State
    recording,
    recordedChunks,
    stream,
    isProcessing,
    isTranscribing,
    transcription,
    showTranscription,
    analysisResults,
    error,
    analysisProgress,

    // Actions
    startRecording,
    stopRecording,
    saveRecording,
    transcribeRecording,
    processAnalysis,
    handleTranscriptionEdit,
    handleTranscriptionSubmit,
    handleTranscriptionCancel,
  };
};
