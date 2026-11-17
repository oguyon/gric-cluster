/**
 * @file gric_milk_unit_helpers.h
 * @brief Shared inline helpers for GRIC MILK_RUNNABLE streaming units.
 */

#ifndef GRIC_MILK_UNIT_HELPERS_H
#define GRIC_MILK_UNIT_HELPERS_H

#include "milk_runnable.h"
#include <stdint.h>
#include <string.h>

/**
 * gric_milk_configure_trigger() - Configure processinfo and local-FPS stream triggers.
 * @self:       Active MILK_RUNNABLE descriptor.
 * @in_name:    Input ImageStreamIO trigger stream name.
 * @max_frames: Maximum frame limit configured in FPS (0 = unbounded).
 */
static inline void gric_milk_configure_trigger(
    const MILK_RUNNABLE *self,
    const char          *in_name,
    uint64_t             max_frames)
{
    strncpy(self->rt->cmddata.cmdsettings->triggerstreamname, in_name,
            STRINGMAXLEN_IMAGE_NAME - 1);
    self->rt->cmddata.cmdsettings->triggerstreamname[STRINGMAXLEN_IMAGE_NAME - 1] = '\0';
    self->rt->cmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    if (dcfpsptr != NULL && dcfpsptr->md->name[0] == '_' &&
        dcfpsptr->cmdset.triggerstreamname[0] == '\0')
    {
        strncpy(dcfpsptr->cmdset.triggerstreamname, in_name,
                STRINGMAXLEN_IMAGE_NAME - 1);
        dcfpsptr->cmdset.triggerstreamname[STRINGMAXLEN_IMAGE_NAME - 1] = '\0';
        dcfpsptr->cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        if (dcfpsptr->cmdset.triggermodeptr != NULL)
        {
            *dcfpsptr->cmdset.triggermodeptr = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        }
        if (dcfpsptr->cmdset.procinfo_loopcntMax == 1 && max_frames != 1)
        {
            long target_loops = (max_frames > 0) ? (long)max_frames : -1;
            dcfpsptr->cmdset.procinfo_loopcntMax = target_loops;
            if (dcfpsptr->cmdset.procinfo_loopcntMax_ptr != NULL)
            {
                *dcfpsptr->cmdset.procinfo_loopcntMax_ptr = target_loops;
            }
        }
    }
}

/**
 * gric_milk_ensure_sem_trigger() - Ensure semaphore trigger wait is bound on input stream.
 * @processinfo: Active PROCESSINFO handle (may be NULL).
 * @in_im:       Connected input IMAGE pointer.
 */
static inline void gric_milk_ensure_sem_trigger(
    PROCESSINFO *processinfo,
    IMAGE       *in_im)
{
    if (processinfo != NULL &&
        (processinfo->triggerstreamID < 0 ||
         processinfo->triggermode != PROCESSINFO_TRIGGERMODE_SEMAPHORE))
    {
        processinfo->triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        processinfo_waitoninputstream_init(processinfo, in_im,
                                           PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    }
}

/**
 * gric_milk_instance_name() - Resolve active FPS or processinfo instance name.
 * @fallback:    Default name when FPS/processinfo names are unset.
 * @processinfo: Active PROCESSINFO handle (may be NULL).
 *
 * Return: Non-NULL instance name string.
 */
static inline const char *gric_milk_instance_name(
    const char        *fallback,
    const PROCESSINFO *processinfo)
{
    if (dcfpsptr != NULL && dcfpsptr->md != NULL && dcfpsptr->md->name[0] != '\0')
    {
        return dcfpsptr->md->name;
    }
    if (processinfo != NULL && processinfo->name[0] != '\0')
    {
        return processinfo->name;
    }
    return fallback;
}

#endif /* GRIC_MILK_UNIT_HELPERS_H */
