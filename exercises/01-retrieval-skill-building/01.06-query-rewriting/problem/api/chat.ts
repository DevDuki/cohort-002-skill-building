import { google } from '@ai-sdk/google';
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateObject,
  streamText,
  type UIMessage,
} from 'ai';
import { ollama } from 'ai-sdk-ollama';
import { z } from 'zod';
import { searchEmails } from './search.ts';

const model = ollama('gpt-oss:20b', {
  think: 'low',
});

export const POST = async (req: Request): Promise<Response> => {
  const body: { messages: UIMessage[] } = await req.json();
  const { messages } = body;

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      // TODO: Change the generateObject call so that it generates a search query in
      // addition to the keywords. This will be used for semantic search, which will be a
      // big improvement over passing the entire conversation history.
      const keywords = await generateObject({
        model,
        system: `You are a helpful email assistant, able to search emails for information.
          Your job is to generate a list of keywords which will be used to search the emails.
          Additionally, suggest a search query, which can be used for a more semantic search algorithm. The query
          can bet a bit more general than the keywords, because it will be used with embeddings rather than keyword
          matching.
        `,
        schema: z.object({
          keywords: z
            .array(z.string())
            .describe(
              'A list of keywords to search the emails with. Use these for exact terminology.',
            ),
          searchQuery: z.string().describe('A search query to search emails in a semantic context, which will be' +
            ' used for embeddings')
        }),
        messages: await convertToModelMessages(messages),
      });

      console.dir(keywords.object, { depth: null });

      const searchResults = await searchEmails({
        keywordsForBM25: keywords.object.keywords,
        embeddingsQuery: keywords.object.searchQuery,
      });

      const topSearchResults = searchResults.slice(0, 5);

      console.log(
        topSearchResults.map((result) => result.email.id),
      );

      const emailSnippets = [
        '## Email Snippets',
        ...topSearchResults.map((result, i) => {
          const from = result.email?.from || 'unknown';
          const to = result.email?.to || 'unknown';
          const subject =
            result.email?.subject || `email-${i + 1}`;
          const body = result.email?.body || '';

          return [
            `### 📧 Email ${i + 1}: [${subject}](#${subject.replace(/[^a-zA-Z0-9]/g, '-')})`,
            `**From:** ${from}`,
            `**To:** ${to}`,
            body,
            '---',
          ].join('\n\n');
        }),
        '## Instructions',
        "Based on the emails above, please answer the user's question. Always cite your sources using the email subject in markdown format.",
      ].join('\n\n');

      const answer = streamText({
        model,
        system: `You are a helpful email assistant that answers questions based on email content.
          You should use the provided emails to answer questions accurately.
          ALWAYS cite sources using markdown formatting with the email subject as the source.
          Be concise but thorough in your explanations.
        `,
        messages: [
          ...await convertToModelMessages(messages),
          {
            role: 'user',
            content: emailSnippets,
          },
        ],
      });

      writer.merge(answer.toUIMessageStream());
    },
  });

  return createUIMessageStreamResponse({
    stream,
  });
};
