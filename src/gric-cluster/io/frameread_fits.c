/**
 * @file frameread_fits.c
 * @brief FITS format reader implementation.
 */

#include "frameread_internal.h"

#ifdef USE_CFITSIO
#include <fitsio.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * init_fits() - Initialize the FITS format frame reader.
 * @filename: Path to the FITS image/cube file.
 *
 * Opens the FITS file using CFITSIO, queries the image dimensions, sets the width, height,
 * and number of frames, and resets the frame index.
 *
 * Return: 0 on success, or -1 on failure.
 */
int init_fits(
    char *filename)
{
    int status = 0;

    if (fits_open_file(&fptr, filename, READONLY, &status))
    {
        fits_report_error(stderr, status);
        return -1;
    }

    {
        /* Query FITS dimensions */
        int naxis;
        long naxes[3] = {0};

        if (fits_get_img_dim(fptr, &naxis, &status) ||
            fits_get_img_size(fptr, 3, naxes, &status))
        {
            fits_report_error(stderr, status);
            return -1;
        }

        if (naxis == 3)
        {
            frame_width = naxes[0];
            frame_height = naxes[1];
            num_frames = naxes[2];
        }
        else if (naxis == 2)
        {
            frame_width = naxes[0];
            frame_height = naxes[1];
            num_frames = 1;
        }
        else
        {
            fprintf(stderr, "Error: Input FITS must be 2D or 3D.\n");
            return -1;
        }
    }

    current_frame_idx = 0;
    return 0;
}

/**
 * getframe_fits() - Retrieve a frame from the FITS file.
 * @frame_struct: Pointer to the Frame struct to populate.
 * @index:        Zero-based index of the frame to retrieve.
 *
 * Reads pixel values from the FITS file at the given frame index and converts them to
 * double-precision floats in the frame data buffer.
 *
 * Return: 0 on success, or -1 on failure.
 */
int getframe_fits(
    Frame *frame_struct,
    long   index)
{
    int status = 0;
    long nelements = frame_width * frame_height;
    long fpixel[3] = {1, 1, index + 1};

    if (fits_read_pix(fptr, TDOUBLE, fpixel, nelements, NULL, frame_struct->data, NULL,
                      &status))
    {
        fits_report_error(stderr, status);
        return -1;
    }

    return 0;
}

/**
 * close_fits() - Close resources associated with the FITS frame reader.
 */
void close_fits(void)
{
    if (fptr != NULL)
    {
        int status = 0;
        fits_close_file(fptr, &status);
        fptr = NULL;
    }
}

/**
 * reset_fits() - Reset the FITS reader position.
 */
void reset_fits(void)
{
    /* FITS reader does not have sequential file pointer state to reset */
}

#endif // USE_CFITSIO
