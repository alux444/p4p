import express from "express";
import { callAgent } from "../utils/call-agent";
import { AgentId } from "../types/agents";
import { getInputString } from "../utils/util";
import { QuestionAndAnswer } from "../types/mocks";
import { tenValues } from "../mocks/responses/10";

export const responseContentRouter = express.Router();

responseContentRouter .post("/", async (req, res) => {
  try {
    const agentId: AgentId = "response-content-analysis";
    const qAndA: QuestionAndAnswer = tenValues[2];
    const inputString = getInputString(qAndA);
    await callAgent(agentId, inputString);
    res.status(200).send("Response content analysis initiated successfully.");
  } catch (error) {
    console.error("Error in response content analysis:", error);
    res.status(500).send("Failed to initiate response content analysis.");
  }
});
