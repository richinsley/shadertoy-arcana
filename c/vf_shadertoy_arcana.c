/*
 * Copyright (c) 2025 Rich Insley
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Shadertoy FFMpeg Arcana filter
 * https://github.com/pygfx/shadertoy
 */

#include <libavfilter/avfilter.h>
#include <libavfilter/filters.h>
#include <libavfilter/formats.h>
#include <libavprivate/libavfilter/video.h>
#include <libavutil/imgutils.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/opt.h>
#include <libavutil/lfg.h>
#include <libavutil/random_seed.h>
#include <float.h>
#include <math.h>
#include "libshadertoyarcana_go.h"

typedef struct ShadertoyContext {
    const AVClass *class;
    int w, h;
    int type;
    AVRational frame_rate;
    uint64_t pts;
    int frame_index;
    uint64_t py_context;
    char *shaderid;
    char *apikey;
} ShadertoyContext;

#define OFFSET(x) offsetof(ShadertoyContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption shadertoy_options[] = {
    {"size", "set frame size", OFFSET(w),          AV_OPT_TYPE_IMAGE_SIZE, {.str="640x480"}, 0,          0, FLAGS },
    {"s",    "set frame size", OFFSET(w),          AV_OPT_TYPE_IMAGE_SIZE, {.str="640x480"}, 0,          0, FLAGS },
    {"rate", "set frame rate", OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str="30"},      0,    INT_MAX, FLAGS },
    {"r",    "set frame rate", OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str="30"},      0,    INT_MAX, FLAGS },
    {"shaderid", "set shader ID", OFFSET(shaderid), AV_OPT_TYPE_STRING, {.str="XsBXWt"},     0,          0, FLAGS },
    {"apikey", "set API key", OFFSET(apikey), AV_OPT_TYPE_STRING, {.str=""},     0,          0, FLAGS },
    {NULL},
};

AVFILTER_DEFINE_CLASS(shadertoy);

static int shadertoy_config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    FilterLink *l = ff_filter_link(outlink);
    ShadertoyContext *s = ctx->priv;

    if (av_image_check_size(s->w, s->h, 0, ctx) < 0)
        return AVERROR(EINVAL);

    outlink->w = s->w;
    outlink->h = s->h;
    outlink->time_base = av_inv_q(s->frame_rate);
    outlink->sample_aspect_ratio = (AVRational) {1, 1};

    return 0;
}

static int shadertoy_request_frame(AVFilterLink *link)
{
    ShadertoyContext *s = link->src->priv;
    AVFrame *frame = ff_get_video_buffer(link, s->w, s->h);

    if (!frame)
        return AVERROR(ENOMEM);

    frame->sample_aspect_ratio = (AVRational) {1, 1};
    frame->pts = s->pts++;
    frame->duration = 1;

    // render the shadertoy to the frame
    float time = (float)s->frame_index * s->frame_rate.den / s->frame_rate.num;
    uint8_t* data = (uint8_t *)renderShadertoy(s->py_context, time);
    s->frame_index++;
    if(data == 0) {
        av_frame_free(&frame);
        return AVERROR(EINVAL);
    }

    // Setup source data pointers and linesizes
    const uint8_t *src_data[4] = { data, NULL, NULL, NULL };
    const int src_linesize[4] = { s->w * 4, 0, 0, 0 };

    av_image_copy(frame->data, frame->linesize,
                 src_data, src_linesize,
                 AV_PIX_FMT_0BGR32, s->w, s->h);


    return ff_filter_frame(link, frame);
}

static const AVFilterPad shadertoy_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .request_frame = shadertoy_request_frame,
        .config_props  = shadertoy_config_output,
    },
};

static int shadertoy_preinit(AVFilterContext *ctx)
{
    ShadertoyContext *s = ctx->priv;

    generatePythonEnv();

    return 0;
}

static int shadertoy_init(AVFilterContext *ctx)
{
    ShadertoyContext *s = ctx->priv;

    // set up the Python REPL context for shadertoy
    if(s->apikey == 0 || s->apikey[0] == 0) {
        av_log(ctx, AV_LOG_ERROR, "Shadertoy API key must be set\n");
        return AVERROR(EINVAL);
    }
    s->py_context = createShadertoyContext(s->w, s->h, s->shaderid, s->apikey);
    if(s->py_context == 0) {
        return AVERROR(EINVAL);
    }
    
    return 0;
}

static void shadertoy_uninit(AVFilterContext *ctx)
{
    ShadertoyContext *s = ctx->priv;
    av_freep(&s->shaderid);
    if(s->py_context != 0) {
        closeShadertoyContext(s->py_context);
    }
}

static int shadertoy_query_formats(AVFilterContext *ctx)
{
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_0BGR32, AV_PIX_FMT_NONE };

    return ff_set_common_formats_from_list(ctx, pix_fmts);
}

const AVFilter ff_vsrc_shadertoy = {
    .name          = "shadertoy",
    .description   = "Shadertoy implemented in Python",
    .priv_size     = sizeof(ShadertoyContext),
    .priv_class    = &shadertoy_class,
    .inputs        = NULL,
    .preinit       = shadertoy_preinit,
    .init          = shadertoy_init,
    .uninit        = shadertoy_uninit,
    FILTER_OUTPUTS(shadertoy_outputs),
    FILTER_QUERY_FUNC(shadertoy_query_formats),
    .flags         = 0 /* AVFILTER_FLAG_SLICE_THREADS */,
};
