"""Chhat LLM enhancement layer.

Our own models (BiLSTM address parser, v2/v3 rooftop segmenters) are the
product core. This module is an *enhancement* layer: plain-language
explanations and Q&A powered by an external chat model via OpenRouter.

The API key is read ONLY from the OPENROUTER_API_KEY environment variable.
It is never hardcoded, never logged, and never committed.
"""
