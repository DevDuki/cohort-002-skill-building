import { google } from '@ai-sdk/google';
import { generateObject, generateText, Output } from 'ai';
import { z } from 'zod';
import { searchViaBM25 } from './bm25.ts';
import { searchChunksViaEmbeddings } from './embeddings.ts';
import { createChunks, reciprocalRankFusion } from './utils.ts';

export type RerankStatus =
  | 'approved'
  | 'rejected'
  | 'not-passed';

export type ChunkWithScores = {
  chunk: string;
  bm25Score: number;
  embeddingScore: number;
  rrfScore: number;
  rerankStatus: RerankStatus;
  rerankOrder?: number; // Position in reranker's output (lower = more relevant)
};

export const searchChunks = async (opts: {
  keywordsForBM25?: string[];
  embeddingsQuery?: string;
  rerankCount?: number;
}): Promise<ChunkWithScores[]> => {
  const chunks = await createChunks();
  const chunkTexts = chunks.map((c) => c.content);

  const bm25SearchResults =
    opts.keywordsForBM25 && opts.keywordsForBM25.length > 0
      ? await searchViaBM25(chunkTexts, opts.keywordsForBM25)
      : [];

  const embeddingsSearchResults = opts.embeddingsQuery
    ? await searchChunksViaEmbeddings(
        chunks,
        opts.embeddingsQuery,
      )
    : [];

  const rrfResults = reciprocalRankFusion([
    embeddingsSearchResults,
    bm25SearchResults,
  ]);

  // Create maps for quick lookup of individual scores
  const bm25Map = new Map(
    bm25SearchResults.map((r) => [r.chunk, r.score]),
  );
  const embeddingMap = new Map(
    embeddingsSearchResults.map((r) => [r.chunk, r.score]),
  );

  // If no reranking needed, return all chunks with 'not-passed' status
  if (!opts.rerankCount || opts.rerankCount === 0) {
    return rrfResults.map((result) => ({
      chunk: result.chunk,
      bm25Score: bm25Map.get(result.chunk) || 0,
      embeddingScore: embeddingMap.get(result.chunk) || 0,
      rrfScore: result.score,
      rerankStatus: 'not-passed' as RerankStatus,
    }));
  }

  // Take top N results for reranking
  const topResultsForReranking = rrfResults.slice(
    0,
    opts.rerankCount,
  );

  // Assign IDs to chunks for reranking
  const topResultsWithId = topResultsForReranking.map(
    (result, i) => ({
      ...result,
      id: i,
    }),
  );

  const topResultsAsMap = new Map(
    topResultsWithId.map((result) => [result.id, result]),
  );

  const chunksWithId = topResultsWithId
    .map((result) =>
      [
        `## ID: ${result.id}`,
        `<content>`,
        result.chunk,
        `</content>`,
      ].join('\n\n'),
    )
    .join('\n\n');

  const searchQuery = [
    opts.keywordsForBM25?.join(' '),
    opts.embeddingsQuery,
  ]
    .filter(Boolean)
    .join(' ');

  // TODO: Call generateObject to generate an array of IDs
  // of the most relevant chunks, based on the user's search query.
  // You should tell the LLM to return only the IDs, not the full chunks.
  // You should also tell the LLM to be selective and only include chunks
  // that are genuinely helpful for answering the question.
  // If a chunk is only tangentially related or not relevant,
  // exclude its ID.
  const rerankedResults = await generateText({
    model: google('gemini-2.0-flash-lite'),
    output: Output.array({
      element: z.number().describe(`The relevant chunk's ID`),
    }),
    system: `
      You are a search result reranker. Your job is to analyze a list of chunks from a TypeScript book and return only the IDs of the most relevant chunks for answering the user's question.
      
      Given a list of chunks with their IDs and content, you should:
      1. Evaluate how relevant each chunk is to the user's search query
      2. Return only the IDs of the most relevant chunks
      
      You should be selective and only include chunks that are genuinely helpful for answering the question. If a chunk is only tangentially related or not relevant, exclude its ID.
      
      Return the IDs as a simple array of numbers.
    `,
    prompt: `
      Search query:
      ${searchQuery}
    
      Available chunks:
      ${chunksWithId}
    
      Return only the IDs of the most relevant chunks for the user's search query.
    `,
  });

  const approvedChunkIds = rerankedResults.output;

  // Create order map from reranker results
  const rerankOrderMap = new Map(
    approvedChunkIds.map((id, index) => [
      topResultsAsMap.get(id)?.chunk,
      index,
    ]),
  );

  const approvedChunkSet = new Set(
    approvedChunkIds
      .map((id) => topResultsAsMap.get(id)?.chunk)
      .filter((chunk) => chunk !== undefined),
  );

  const passedToRerankerSet = new Set(
    topResultsForReranking.map((r) => r.chunk),
  );

  // Combine all scores and rerank status
  const chunksWithStatus = rrfResults.map((result) => {
    let rerankStatus: RerankStatus;
    let rerankOrder: number | undefined;

    if (approvedChunkSet.has(result.chunk)) {
      rerankStatus = 'approved';
      rerankOrder = rerankOrderMap.get(result.chunk);
    } else if (passedToRerankerSet.has(result.chunk)) {
      rerankStatus = 'rejected';
    } else {
      rerankStatus = 'not-passed';
    }

    return {
      chunk: result.chunk,
      bm25Score: bm25Map.get(result.chunk) || 0,
      embeddingScore: embeddingMap.get(result.chunk) || 0,
      rrfScore: result.score,
      rerankStatus,
      rerankOrder,
    };
  });

  // Sort by rerank status and order:
  // 1. Approved chunks (sorted by reranker order)
  // 2. Rejected chunks (sorted by RRF score)
  // 3. Not-passed chunks (sorted by RRF score)
  return chunksWithStatus.sort((a, b) => {
    // Approved chunks come first, sorted by rerank order
    if (
      a.rerankStatus === 'approved' &&
      b.rerankStatus === 'approved'
    ) {
      return (a.rerankOrder ?? 0) - (b.rerankOrder ?? 0);
    }
    if (a.rerankStatus === 'approved') return -1;
    if (b.rerankStatus === 'approved') return 1;

    // Rejected chunks come next, sorted by RRF score
    if (
      a.rerankStatus === 'rejected' &&
      b.rerankStatus === 'rejected'
    ) {
      return b.rrfScore - a.rrfScore;
    }
    if (a.rerankStatus === 'rejected') return -1;
    if (b.rerankStatus === 'rejected') return 1;

    // Not-passed chunks come last, sorted by RRF score
    return b.rrfScore - a.rrfScore;
  });
};
