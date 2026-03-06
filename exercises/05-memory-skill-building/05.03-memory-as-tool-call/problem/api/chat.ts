import { google } from '@ai-sdk/google';
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  generateId,
  stepCountIs,
  streamText,
  tool,
  type UIMessage,
} from 'ai';
import { z } from 'zod';
import {
  loadMemories,
  saveMemories,
  deleteMemory,
  updateMemory,
  type DB,
} from './memory-persistence.ts';

export type MyMessage = UIMessage<unknown, {}>;

const formatMemory = (memory: DB.MemoryItem) => {
  return [
    `Memory: ${memory.memory}`,
    `ID: ${memory.id}`,
    `Created At: ${memory.createdAt}`,
  ].join('\n');
};

export const POST = async (req: Request): Promise<Response> => {
  const body: { messages: MyMessage[] } = await req.json();
  const { messages } = body;

  const memories = await loadMemories();

  const memoriesText = memories.map(formatMemory).join('\n\n');

  const result = streamText({
    model: google('gemini-2.5-flash-lite'),
    system: `You are a helpful assistant that can answer questions and help with tasks.

    The date is ${new Date().toISOString().split('T')[0]}.

    You have access to the following memories:

    <memories>
    ${memoriesText}
    </memories>

    When users share new personal information, contradict previous information, or ask you to remember/forget things, use the manageMemories tool to update the memory system.
    `,
    messages: await convertToModelMessages(messages),
    // TODO: Add the manageMemories tool
    // The tool should have three parameters:
    // - updates: array of objects with { id: string, memory: string }
    // - deletions: array of strings (memory IDs to delete)
    // - additions: array of strings (new memories to add)
    // In the execute function, perform the actual memory operations
    tools: {
      manageMemories: tool({
        description:
          'Manage user memories by adding new ones, updating existing ones, or deleting outdated/incorrect ones. Call this when the user shares personal information, contradicts previous statements, or explicitly asks to remember/forget something.',
        inputSchema: z.object({
          updates: z
            .array(
              z.object({
                id: z
                  .string()
                  .describe(
                    'The ID of the existing memory to update',
                  ),
                memory: z
                  .string()
                  .describe('The updated memory content'),
              }),
            )
            .describe(
              'Array of existing memories that need to be updated with new information',
            ),
          deletions: z
            .array(z.string())
            .describe(
              'Array of memory IDs that should be deleted (outdated, incorrect, or no longer relevant)',
            ),
          additions: z
            .array(z.string())
            .describe(
              "Array of new memory strings to add to the user's permanent memory",
            ),
        }),
        execute: async ({ updates, deletions, additions }) => {
          console.log('Memory tool called:');
          console.log('Updates:', updates);
          console.log('Deletions:', deletions);
          console.log('Additions:', additions);

          // Filter out deletions that are being updated
          const filteredDeletions = deletions.filter(
            (deletion) =>
              !updates.some((update) => update.id === deletion),
          );

          // Perform updates
          updates.forEach((update) =>
            updateMemory(update.id, {
              memory: update.memory,
              createdAt: new Date().toISOString(),
            }),
          );

          // Perform deletions
          filteredDeletions.forEach((deletion) =>
            deleteMemory(deletion),
          );

          // Perform additions
          saveMemories(
            additions.map((addition) => ({
              id: generateId(),
              memory: addition,
              createdAt: new Date().toISOString(),
            })),
          );

          return {
            success: true,
            message: `Updated ${updates.length} memories, deleted ${filteredDeletions.length} memories, added ${additions.length} new memories.`,
          };
        },
      }),
    },
    // TODO: Add stopWhen with stepCountIs to allow the agent to call tools
    // Use stepCountIs(5) to allow up to 5 generation steps
    stopWhen: stepCountIs(5),
  });

  return createUIMessageStreamResponse({
    stream: result.toUIMessageStream(),
  });
};
