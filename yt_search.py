from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
import csv
from datetime import datetime

YOUTUBE_API_KEY = "AIzaSyB4TeDijZQ2YK5QbL2Gc8tO0eY8-cLCLYs"

# Initialize the YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def search_youtube(query, max_results=5):
    query = "tony seba and " + query
    print(query)
    print("Searching YouTube...")
    try:
        search_response = youtube.search().list(
            q=query,
            type='video',
            part='id,snippet',
            maxResults=max_results
        ).execute()

        videos = []
        for search_result in search_response.get('items', []):
            video = {
                'title': search_result['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                'description': search_result['snippet']['description'],
                'video_id': search_result['id']['videoId']
            }
            videos.append(video)
        return videos
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return []

def calculate_relevance(query, text):
    query_embedding = model.encode([query])
    text_embedding = model.encode([text])
    return cosine_similarity(query_embedding, text_embedding)[0][0]

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {str(e)}")
        return ""

def main():
    query = input("Enter your YouTube search query: ")
    results = []

    youtube_results = search_youtube(query)
    for video in youtube_results:
        transcript = get_transcript(video['video_id'])
        relevance = calculate_relevance(query, video['title'] + " " + video['description'] + " " + transcript)
        results.append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'url': video['url'],
            'title': video['title'],
            'content_type': 'Video',
            'relevance_score': relevance,
            'transcript': transcript[:1000] + '...' if len(transcript) > 1000 else transcript  # Truncate long transcripts
        })

    # Sort results by relevance score
    results.sort(key=lambda x: x['relevance_score'], reverse=True)

    # Save results to CSV
    with open('youtube_search_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'url', 'title', 'content_type', 'relevance_score', 'transcript']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nSearch complete. Results saved to youtube_search_results.csv")

    # Print results
    print("\nYouTube Search Results:")
    for result in results:
        print(f"Date: {result['date']}, Title: {result['title']}, URL: {result['url']}, Relevance Score: {result['relevance_score']:.2f}")
        print(f"Transcript preview: {result['transcript'][:200]}...")  # Print first 200 characters of transcript
        print()

if __name__ == "__main__":
    main()
