-- =============================================================================
-- MULTIMODAL LECTURE SUMMARIZER - DATABASE INITIALIZATION SCRIPT (POSTGRESQL)
-- =============================================================================
-- This script sets up all custom ENUM types, tables, constraints, indexes,
-- foreign keys with ON DELETE CASCADE, and default initial seed data.
-- Compatible with PostgreSQL 14, 15, 16, 17, 18+ (including Neon Serverless & Supabase).
-- =============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Define Custom ENUM Types (Create only if they don't already exist)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'userrole') THEN
        CREATE TYPE userrole AS ENUM ('ADMIN', 'USER');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'videostatus') THEN
        CREATE TYPE videostatus AS ENUM ('PENDING', 'PROCESSING', 'DONE', 'FAILED');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ragstatus') THEN
        CREATE TYPE ragstatus AS ENUM ('PENDING', 'PROCESSING', 'READY', 'FAILED');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'jobtype') THEN
        CREATE TYPE jobtype AS ENUM ('SUMMARIZE', 'QA');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'jobstatus') THEN
        CREATE TYPE jobstatus AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED');
    END IF;
END $$;

-- 3. Create Tables

-- Table: users
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role userrole NOT NULL DEFAULT 'USER',
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    last_login TIMESTAMP WITHOUT TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Table: video_standards (Stores global system limits and upload constraints)
CREATE TABLE IF NOT EXISTS video_standards (
    standard_id SERIAL PRIMARY KEY,
    max_duration INTEGER NOT NULL DEFAULT 3600,
    allowed_formats VARCHAR(255) NOT NULL DEFAULT 'mp4,avi,mkv',
    max_file_size INTEGER NOT NULL DEFAULT 500,
    min_audio_quality DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    allowed_codecs JSON NOT NULL DEFAULT '["h264", "hevc"]',
    min_width INTEGER NOT NULL DEFAULT 640,
    min_height INTEGER NOT NULL DEFAULT 360,
    max_width INTEGER NOT NULL DEFAULT 1920,
    max_height INTEGER NOT NULL DEFAULT 1080
);

-- Table: videos
CREATE TABLE IF NOT EXISTS videos (
    video_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    original_url VARCHAR(2048),
    file_path VARCHAR(1024),
    duration DOUBLE PRECISION,
    language VARCHAR(50) NOT NULL DEFAULT 'en',
    status videostatus NOT NULL DEFAULT 'PENDING',
    rag_status ragstatus NOT NULL DEFAULT 'PENDING',
    title VARCHAR(256),
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Table: video_metadata (1-to-1 relationship with videos)
CREATE TABLE IF NOT EXISTS video_metadata (
    video_id UUID PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    video_codec VARCHAR(50),
    audio_codec VARCHAR(50),
    fps DOUBLE PRECISION NOT NULL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bitrate INTEGER,
    video_source VARCHAR(100) NOT NULL DEFAULT 'upload',
    video_path VARCHAR(1024) NOT NULL DEFAULT ''
);

-- Table: video_scenes (Extracted visual scene segments and keyframes)
CREATE TABLE IF NOT EXISTS video_scenes (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    scene_index INTEGER NOT NULL,
    start_seconds DOUBLE PRECISION NOT NULL,
    end_seconds DOUBLE PRECISION NOT NULL,
    start_timecode VARCHAR(20) NOT NULL,
    end_timecode VARCHAR(20) NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    keyframe_path VARCHAR(1024) NOT NULL,
    keyframe_url VARCHAR(2048) NOT NULL,
    caption TEXT NOT NULL,
    script TEXT,
    detected_objects_json JSON,
    object_counts_json JSON
);

-- Table: jobs (Background processing pipeline and Celery tasks tracking)
CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    job_type jobtype NOT NULL DEFAULT 'SUMMARIZE',
    status jobstatus NOT NULL DEFAULT 'PENDING',
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE,
    error_log TEXT
);

-- Table: summaries (Generated multimodal summaries and chapter segments)
CREATE TABLE IF NOT EXISTS summaries (
    summary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL UNIQUE REFERENCES videos(video_id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    chapters_json JSON NOT NULL,
    keyframes_json JSON NOT NULL,
    key_takeaways_json JSON,
    objects_summary_json JSON,
    transcript_text TEXT NOT NULL,
    model_used TEXT NOT NULL,
    processing_time DOUBLE PRECISION NOT NULL
);

-- Table: evaluation_logs (Ground-truth benchmark logs for WER/CER/ROUGE/OD metrics)
CREATE TABLE IF NOT EXISTS evaluation_logs (
    eval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(video_id) ON DELETE CASCADE,
    wer DOUBLE PRECISION,
    cer DOUBLE PRECISION,
    rouge_1 DOUBLE PRECISION,
    rouge_2 DOUBLE PRECISION,
    rouge_l DOUBLE PRECISION,
    bertscore_f1 DOUBLE PRECISION,
    od_precision DOUBLE PRECISION,
    od_recall DOUBLE PRECISION,
    od_map50 DOUBLE PRECISION,
    od_map50_95 DOUBLE PRECISION,
    reference_transcript TEXT,
    reference_summary TEXT,
    details_json JSON,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Table: qa_logs (User interactive Q&A questions, answers, and retrieved RAG context)
CREATE TABLE IF NOT EXISTS qa_logs (
    qa_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieved_chunks JSON,
    asked_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Table: system_stats (Daily aggregated analytics and usage counters)
CREATE TABLE IF NOT EXISTS system_stats (
    stat_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    total_users INTEGER NOT NULL DEFAULT 0,
    new_users INTEGER NOT NULL DEFAULT 0,
    total_videos_processed INTEGER NOT NULL DEFAULT 0,
    total_jobs_run INTEGER NOT NULL DEFAULT 0
);

-- 4. Create Indexes for Query Optimization
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_rag_status ON videos(rag_status);
CREATE INDEX IF NOT EXISTS idx_video_scenes_video_id ON video_scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_video_scenes_index ON video_scenes(video_id, scene_index);
CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON summaries(video_id);
CREATE INDEX IF NOT EXISTS idx_qa_logs_video_id ON qa_logs(video_id);
CREATE INDEX IF NOT EXISTS idx_qa_logs_user_id ON qa_logs(user_id);

-- 5. Seed Default Data (Idempotent: only inserts if not already present)

-- Default Video Standard (Limits: 3600s / 60 mins, mp4/avi/mkv, 500MB, h264/hevc, 640x360 to 1920x1080)
INSERT INTO video_standards (standard_id, max_duration, allowed_formats, max_file_size, min_audio_quality, allowed_codecs, min_width, min_height, max_width, max_height)
SELECT 1, 3600, 'mp4,avi,mkv', 500, 0.0, '["h264", "hevc"]'::json, 640, 360, 1920, 1080
WHERE NOT EXISTS (SELECT 1 FROM video_standards WHERE standard_id = 1);

-- Default Admin User: hungphitran.22@gmail.com / AdminPass123@
INSERT INTO users (user_id, email, password_hash, role, is_active)
SELECT 
    'a0000000-0000-0000-0000-000000000001'::uuid, 
    'hungphitran.22@gmail.com', 
    '$2b$12$WP1VXXO7NHn/ORqAOfb9kOvF/h6DpE9etgljTom9XXPyfgxSFBhCa', 
    'ADMIN', 
    TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'hungphitran.22@gmail.com');

-- Default Regular User: nguyen.van.a@gmail.com / UserPass123@
INSERT INTO users (user_id, email, password_hash, role, is_active)
SELECT 
    'b0000000-0000-0000-0000-000000000002'::uuid, 
    'nguyen.van.a@gmail.com', 
    '$2b$12$DLuiCwimJ5rrmzuSLEGT8O8JL71zl7eDwpuE2cNJUMVxxQP66RCKq', 
    'USER', 
    TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'nguyen.van.a@gmail.com');
