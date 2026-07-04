/**
 * @file frameread_ffmpeg.c
 * @brief FFmpeg MP4 format reader implementation.
 */

#include "frameread_internal.h"

#ifdef USE_FFMPEG
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * init_mp4() - Initialize the FFmpeg MP4 format frame reader.
 * @filename: Path to the MP4 video file.
 *
 * Opens the video file, discovers the video stream, allocates and initializes the decoder,
 * allocates structures, computes the frame dimensions, and prepares the scaler context.
 *
 * Return: 0 on success, or -1 on failure.
 */
int init_mp4(
    char *filename)
{
    if (avformat_open_input(&fmt_ctx, filename, NULL, NULL) < 0)
    {
        fprintf(stderr, "Could not open video file %s\n", filename);
        return -1;
    }

    if (avformat_find_stream_info(fmt_ctx, NULL) < 0)
    {
        fprintf(stderr, "Could not find stream information\n");
        goto cleanup_format;
    }

    video_stream_idx = -1;
    for (unsigned int ii = 0; ii < fmt_ctx->nb_streams; ii++)
    {
        if (fmt_ctx->streams[ii]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_idx = (int)ii;
            break;
        }
    }

    if (video_stream_idx == -1)
    {
        fprintf(stderr, "Could not find video stream\n");
        goto cleanup_format;
    }

    {
        AVCodecParameters *codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
        const AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
        if (codec == NULL)
        {
            fprintf(stderr, "Codec not found\n");
            goto cleanup_format;
        }

        dec_ctx = avcodec_alloc_context3(codec);
        if (dec_ctx == NULL)
        {
            fprintf(stderr, "Could not allocate video codec context\n");
            goto cleanup_format;
        }

        if (avcodec_parameters_to_context(dec_ctx, codecpar) < 0)
        {
            fprintf(stderr, "Failed to copy codec parameters to decoder context\n");
            goto cleanup_codec;
        }

        if (avcodec_open2(dec_ctx, codec, NULL) < 0)
        {
            fprintf(stderr, "Could not open codec\n");
            goto cleanup_codec;
        }
    }

    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (frame == NULL || pkt == NULL)
    {
        fprintf(stderr, "Could not allocate frame or packet\n");
        goto cleanup_frame_pkt;
    }

    /* Interleaved RGB */
    frame_width = dec_ctx->width * 3;
    frame_height = dec_ctx->height;

    if (fmt_ctx->streams[video_stream_idx]->nb_frames > 0)
    {
        num_frames = fmt_ctx->streams[video_stream_idx]->nb_frames;
    }
    else
    {
        /* Fallback estimate */
        double duration = (double)fmt_ctx->duration / AV_TIME_BASE;
        double fps = av_q2d(fmt_ctx->streams[video_stream_idx]->avg_frame_rate);
        if (duration > 0.0 && fps > 0.0)
        {
            num_frames = (long)(duration * fps);
        }
        else
        {
            num_frames = 10000;
        }
        printf("Warning: Could not determine exact frame count. Using estimated %ld\n",
               num_frames);
    }

    is_mp4_mode = 1;

    /* Prepare scaler for RGB24 */
    sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
                             dec_ctx->width, dec_ctx->height, AV_PIX_FMT_RGB24,
                             SWS_BILINEAR, NULL, NULL, NULL);
    if (sws_ctx == NULL)
    {
        fprintf(stderr, "Could not initialize sws context\n");
        goto cleanup_frame_pkt;
    }

    return 0;

cleanup_frame_pkt:
    if (frame != NULL)
    {
        av_frame_free(&frame);
        frame = NULL;
    }
    if (pkt != NULL)
    {
        av_packet_free(&pkt);
        pkt = NULL;
    }
cleanup_codec:
    avcodec_free_context(&dec_ctx);
    dec_ctx = NULL;
cleanup_format:
    avformat_close_input(&fmt_ctx);
    fmt_ctx = NULL;
    return -1;
}

/**
 * getframe_mp4() - Retrieve a frame from the MP4 video stream.
 * @frame_struct: Pointer to the Frame struct to populate.
 * @index:        Zero-based index of the frame to retrieve.
 *
 * Reads packets from the format context, decodes video frames, scales the output to RGB24,
 * and copies the values into the frame's data buffer.
 *
 * Return: 0 on success, or -1 on failure.
 */
int getframe_mp4(
    Frame *frame_struct,
    long   index)
{
    long nelements = frame_width * frame_height;

    /* Handle seeking if necessary */
    if (index != internal_mp4_index)
    {
        if (index < internal_mp4_index)
        {
            /* Rewind */
            av_seek_frame(fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(dec_ctx);
            internal_mp4_index = 0;
        }

        /* Fast forward */
        while (internal_mp4_index < index)
        {
            if (av_read_frame(fmt_ctx, pkt) < 0)
            {
                break;
            }
            if (pkt->stream_index == video_stream_idx)
            {
                if (avcodec_send_packet(dec_ctx, pkt) == 0)
                {
                    while (avcodec_receive_frame(dec_ctx, frame) == 0)
                    {
                        internal_mp4_index++;
                    }
                }
            }
            av_packet_unref(pkt);
        }
    }

    /* Read next frame */
    {
        int ret = 0;
        int frame_decoded = 0;
        while (ret >= 0 && !frame_decoded)
        {
            ret = av_read_frame(fmt_ctx, pkt);
            if (ret < 0)
            {
                break;
            }
            if (pkt->stream_index == video_stream_idx)
            {
                if (avcodec_send_packet(dec_ctx, pkt) == 0)
                {
                    if (avcodec_receive_frame(dec_ctx, frame) == 0)
                    {
                        frame_decoded = 1;
                        internal_mp4_index++;
                    }
                }
            }
            av_packet_unref(pkt);
        }

        if (!frame_decoded)
        {
            return -1;
        }
    }

    {
        /* Scale frame and copy to destination */
        uint8_t *rgb_data[4] = {NULL};
        int rgb_linesize[4] = {0};

        rgb_data[0] = (uint8_t *)malloc(dec_ctx->width * dec_ctx->height * 3);
        if (rgb_data[0] == NULL)
        {
            return -1;
        }
        rgb_linesize[0] = dec_ctx->width * 3;

        sws_scale(sws_ctx, (const uint8_t *const *)frame->data, frame->linesize, 0,
                  dec_ctx->height, rgb_data, rgb_linesize);

        uint8_t *src = rgb_data[0];
        for (long ii = 0; ii < nelements; ii++)
        {
            frame_struct->data[ii] = (double)src[ii];
        }

        free(rgb_data[0]);
    }

    return 0;
}

/**
 * close_mp4() - Close resources associated with the MP4 frame reader.
 */
void close_mp4(void)
{
    if (dec_ctx != NULL)
    {
        avcodec_free_context(&dec_ctx);
        dec_ctx = NULL;
    }
    if (fmt_ctx != NULL)
    {
        avformat_close_input(&fmt_ctx);
        fmt_ctx = NULL;
    }
    if (frame != NULL)
    {
        av_frame_free(&frame);
        frame = NULL;
    }
    if (pkt != NULL)
    {
        av_packet_free(&pkt);
        pkt = NULL;
    }
    if (sws_ctx != NULL)
    {
        sws_freeContext(sws_ctx);
        sws_ctx = NULL;
    }
    is_mp4_mode = 0;
}

/**
 * reset_mp4() - Reset the MP4 frame reader position to the beginning.
 */
void reset_mp4(void)
{
    if (fmt_ctx != NULL)
    {
        av_seek_frame(fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec_ctx);
        internal_mp4_index = 0;
    }
}

#endif // USE_FFMPEG
