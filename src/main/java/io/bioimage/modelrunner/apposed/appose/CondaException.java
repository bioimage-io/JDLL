/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.apposed.appose;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
public class CondaException
{

	public static class EnvironmentExistsException extends RuntimeException
	{
		private static final long serialVersionUID = -1625119813967214783L;

		/**
		 * Constructs a new exception with {@code null} as its detail message. The cause
		 * is not initialized, and may subsequently be initialized by a call to
		 * {@link #initCause}.
		 */
		public EnvironmentExistsException()
		{
			super();
		}

		/**
		 * Constructs a new exception with the specified detail message. The cause is
		 * not initialized, and may subsequently be initialized by a call to
		 * {@link #initCause}.
		 *
		 * @param msg
		 *            the detail message. The detail message is saved for later
		 *            retrieval by the {@link #getMessage()} method.
		 */
		public EnvironmentExistsException( String msg )
		{
			super( msg );
		}

		/**
		 * Constructs a new exception with the specified detail message and cause.
		 * <p>
		 * Note that the detail message associated with {@code cause} is <i>not</i>
		 * automatically incorporated in this exception's detail message.
		 *
		 * @param message
		 *            the detail message (which is saved for later retrieval by the
		 *            {@link #getMessage()} method).
		 * @param cause
		 *            the cause (which is saved for later retrieval by the
		 *            {@link #getCause()} method). (A <tt>null</tt> value is permitted,
		 *            and indicates that the cause is nonexistent or unknown.)
		 * @since 1.4
		 */
		public EnvironmentExistsException( String message, Throwable cause )
		{
			super( message, cause );
		}

		/**
		 * Constructs a new exception with the specified cause and a detail message of
		 * <tt>(cause==null ? null : cause.toString())</tt> (which typically contains
		 * the class and detail message of <tt>cause</tt>). This constructor is useful
		 * for exceptions that are little more than wrappers for other throwables (for
		 * example, {@link java.security.PrivilegedActionException}).
		 *
		 * @param cause
		 *            the cause (which is saved for later retrieval by the
		 *            {@link #getCause()} method). (A <tt>null</tt> value is permitted,
		 *            and indicates that the cause is nonexistent or unknown.)
		 * @since 1.4
		 */
		public EnvironmentExistsException( Throwable cause )
		{
			super( cause );
		}

		/**
		 * Constructs a new exception with the specified detail message, cause,
		 * suppression enabled or disabled, and writable stack trace enabled or
		 * disabled.
		 *
		 * @param message
		 *            the detail message.
		 * @param cause
		 *            the cause. (A {@code null} value is permitted, and indicates that
		 *            the cause is nonexistent or unknown.)
		 * @param enableSuppression
		 *            whether or not suppression is enabled or disabled
		 * @param writableStackTrace
		 *            whether or not the stack trace should be writable
		 * @since 1.7
		 */
		protected EnvironmentExistsException( String message, Throwable cause,
				boolean enableSuppression,
				boolean writableStackTrace )
		{
			super( message, cause, enableSuppression, writableStackTrace );
		}
	}

}
