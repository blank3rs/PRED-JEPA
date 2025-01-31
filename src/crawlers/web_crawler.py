import os
import time
import logging
import threading
import asyncio
import aiohttp
import psutil
import json
import hashlib
import sqlite3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from queue import Queue
from PIL import Image
import io
import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta

class DFSWebCrawler:
    def __init__(self, max_depth=10, max_threads=None, cache_dir='./crawler_cache'):
        self.max_depth = max_depth
        self.visited = set()
        self.data_lock = threading.Lock()
        self.is_running = True
        self.loop = None
        self.cache_dir = cache_dir
        self.tasks = set()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'images'), exist_ok=True)
        
        # Initialize SQLite cache with optimized settings
        self.init_db()
        
        # Auto-detect optimal thread count based on system resources
        if max_threads is None:
            cpu_count = os.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            network_bandwidth = 50  # Assume 50 Mbps as baseline
            # Calculate optimal concurrent tasks based on resources
            self.max_concurrent = min(
                int(cpu_count * 6),  # More aggressive CPU utilization
                int(memory_gb * 5),  # More memory per task
                int(network_bandwidth * 2),  # Better bandwidth utilization
                150  # Hard cap at 150 tasks
            )
        else:
            self.max_concurrent = max_threads
            
        # Optimize domain rate limiting
        self.domain_locks = {}
        self.domain_lock_lock = threading.Lock()
        self.domain_stats = {}
        
        # Larger queues for better buffering
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.text_queue = Queue(maxsize=int(5000 * memory_gb))  # Increased queue sizes
        self.image_queue = Queue(maxsize=int(2000 * memory_gb))
        
        # Enhanced batch processing settings
        self.batch_size = 20  # Increased batch size
        self.pending_urls = []
        self.processing_timeout = 30  # 30 second timeout for processing
        
        # Metrics initialization
        self.metrics = {
            'start_time': time.time(),
            'pages_crawled': 0,
            'bytes_downloaded': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_hits': 0
        }
        
        # Load visited URLs from cache
        self.load_visited_urls()
        
        logging.info(f"Initialized enhanced DFS crawler with {self.max_concurrent} concurrent tasks and optimized settings")

    def init_db(self):
        """Initialize SQLite database with optimized settings"""
        self.db_path = os.path.join(self.cache_dir, 'crawler_cache.db')
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=50000")  # Increased cache size
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=30000000000")  # 30GB mmap
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS pages
                        (url TEXT PRIMARY KEY, content TEXT, last_crawled TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS visited_urls
                        (url TEXT PRIMARY KEY, timestamp TIMESTAMP)''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_last_crawled ON pages(last_crawled)')
            conn.commit()

    def load_visited_urls(self):
        """Load previously visited URLs from cache"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT url FROM visited_urls')
            self.visited = set(row[0] for row in c.fetchall())
            logging.info(f"Loaded {len(self.visited)} visited URLs from cache")

    def save_visited_url(self, url):
        """Save visited URL to cache"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO visited_urls (url, timestamp) VALUES (?, ?)',
                     (url, datetime.now()))
            conn.commit()

    def get_cached_page(self, url, max_age_hours=24):
        """Get page from cache if it exists and is not too old"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT content, last_crawled FROM pages WHERE url = ?', (url,))
            result = c.fetchone()
            if result:
                content, last_crawled = result
                # Check if cache is less than max_age_hours old
                if datetime.now() - datetime.fromisoformat(last_crawled) < timedelta(hours=max_age_hours):
                    return content
        return None

    def cache_page(self, url, content):
        """Cache page content"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO pages (url, content, last_crawled) VALUES (?, ?, ?)',
                     (url, content, datetime.now()))
            conn.commit()

    def get_adaptive_delay(self, domain):
        """Get adaptive delay with optimized settings"""
        stats = self.domain_stats.get(domain, {'errors': 0, 'success': 0})
        base_delay = 0.2  # Reduced base delay
        
        # More aggressive delay adjustment
        error_factor = 1 + (stats['errors'] * 0.2)  # Reduced penalty
        success_factor = max(0.1, 1 - (stats['success'] * 0.25))  # More reduction for success
        
        return base_delay * error_factor * success_factor

    def delete_cached_image(self, img_url):
        """Delete cached image file"""
        try:
            img_hash = hashlib.md5(img_url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, 'images', f'{img_hash}.jpg')
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as e:
            logging.error(f"Error deleting cached image {img_url}: {e}")

    async def process_image(self, session, img_url):
        """Process and cache images with cleanup"""
        try:
            # Generate cache path
            img_hash = hashlib.md5(img_url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, 'images', f'{img_hash}.jpg')
            
            # Check cache first
            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert('RGB')
                # Delete the cached image after loading
                self.delete_cached_image(img_url)
                return image
            
            async with session.get(img_url, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    
                    # Cache temporarily and then delete after processing
                    image.save(cache_path, 'JPEG', quality=85)
                    processed_image = Image.open(cache_path).convert('RGB')
                    self.delete_cached_image(img_url)
                    return processed_image
                    
        except Exception as e:
            logging.error(f"Error processing image {img_url}: {e}")
            # Clean up any failed cached image
            self.delete_cached_image(img_url)
            return None

    async def crawl_url_dfs(self, url, depth=0):
        """Optimized crawling function with cleanup"""
        if not self.is_running or depth > self.max_depth or url in self.visited:
            return

        domain = urlparse(url).netloc
        domain_lock = self.get_domain_lock(domain)

        async with domain_lock:
            try:
                with self.data_lock:
                    if url in self.visited:
                        return
                    self.visited.add(url)
                    self.save_visited_url(url)

                # Check cache with shorter expiry
                cached_content = self.get_cached_page(url, max_age_hours=12)
                if cached_content:
                    self.process_cached_content(url, cached_content, depth)
                    return

                # Reduced delay
                delay = self.get_adaptive_delay(domain)
                await asyncio.sleep(delay)

                # Optimized network settings
                timeout = aiohttp.ClientTimeout(total=20, connect=5)
                connector = aiohttp.TCPConnector(limit_per_host=10)
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    try:
                        async with session.get(url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                        }) as response:
                            if response.status != 200:
                                self.update_domain_stats(domain, success=False)
                                return

                            content_type = response.headers.get('content-type', '').lower()
                            if 'text/html' not in content_type:
                                return

                            content = await response.text()
                            self.metrics['bytes_downloaded'] += len(content)
                            self.metrics['pages_crawled'] += 1
                            self.update_domain_stats(domain, success=True)

                            # Process content immediately
                            soup = BeautifulSoup(content, 'html.parser')
                            await asyncio.gather(
                                self.process_text(url, soup, depth),
                                self.process_images(session, soup, url, depth),
                                self.process_links(soup, url, depth)
                            )

                            # Delete the URL from visited_urls after processing
                            self.delete_visited_url(url)

                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        self.update_domain_stats(domain, success=False)
                        logging.error(f"Network error crawling {url}: {e}")

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")

    async def process_text(self, url, soup, depth):
        """Process text content asynchronously"""
        text = self.extract_text(soup)
        if text and len(text.split()) > 50:
            if not self.text_queue.full():
                self.text_queue.put({'url': url, 'text': text, 'depth': depth})

    async def process_images(self, session, soup, base_url, depth):
        """Process images from a page with immediate cleanup"""
        for img in soup.find_all('img'):
            if not self.is_running:
                return
                
            img_url = img.get('src')
            if img_url and self.is_valid_image_url(img_url):
                img_url = urljoin(base_url, img_url)
                processed_image = await self.process_image(session, img_url)
                if processed_image and not self.image_queue.full():
                    self.image_queue.put({
                        'url': img_url,
                        'image': processed_image,
                        'depth': depth
                    })
                    # Clean up the processed image from memory
                    processed_image.close()

    async def process_links(self, soup, base_url, depth):
        """Process and schedule links asynchronously"""
        if depth < self.max_depth:
            links = self.extract_links(soup, base_url)
            for link in links[:50]:  # Limit to top 50 links per page
                if len(self.tasks) < self.max_concurrent and self.is_running:
                    task = asyncio.create_task(self.crawl_url_dfs(link, depth + 1))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)

    def extract_links(self, soup, base_url):
        """Extract and filter links from a page"""
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                if self.is_valid_url(full_url) and self.should_crawl_url(full_url) and full_url not in self.visited:
                    links.append(full_url)
        return links

    async def run_crawler(self, seed_urls):
        """Main crawler coroutine"""
        try:
            # Create initial tasks
            for url in seed_urls:
                if len(self.tasks) < self.max_concurrent:
                    task = asyncio.create_task(self.crawl_url_dfs(url))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
            
            # Wait for all tasks to complete or until stopped
            while self.tasks and self.is_running:
                # Wait for any task to complete with timeout
                done, pending = await asyncio.wait(
                    self.tasks,
                    timeout=1,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        logging.error(f"Task error: {e}")
                    
                    # Create new tasks if we have capacity
                    if len(self.tasks) < self.max_concurrent and self.is_running:
                        # Find unvisited URLs from our visited set's outgoing links
                        unvisited_urls = []
                        for visited_url in self.visited:
                            cached_content = self.get_cached_page(visited_url)
                            if cached_content:
                                soup = BeautifulSoup(cached_content, 'html.parser')
                                links = self.extract_links(soup, visited_url)
                                unvisited_urls.extend(links)
                        
                        # Create new tasks for unvisited URLs
                        for url in unvisited_urls:
                            if len(self.tasks) >= self.max_concurrent:
                                break
                            if url not in self.visited and self.is_running:
                                new_task = asyncio.create_task(self.crawl_url_dfs(url))
                                self.tasks.add(new_task)
                                new_task.add_done_callback(self.tasks.discard)
                
                # Log progress
                if self.tasks:
                    metrics = self.get_enhanced_metrics()
                    logging.info(
                        f"Active tasks: {len(self.tasks)}, "
                        f"Pages crawled: {metrics['pages_crawled']}, "
                        f"Crawl rate: {metrics['crawl_rate']:.2f} pages/sec, "
                        f"Success rate: {metrics['success_rate']:.2%}"
                    )
                
        except Exception as e:
            logging.error(f"Crawler error: {e}")

    def run_async_loop(self, seed_urls):
        """Run the async event loop in the current thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self.run_crawler(seed_urls))
            except asyncio.CancelledError:
                logging.info("Crawler tasks cancelled")
            except Exception as e:
                logging.error(f"Error in crawler execution: {e}")
            
        except Exception as e:
            logging.error(f"Error in crawler async loop: {e}")
        finally:
            try:
                # Cancel remaining tasks
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete with timeout
                if self.tasks:
                    try:
                        self.loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*self.tasks, return_exceptions=True),
                                timeout=5
                            )
                        )
                    except asyncio.TimeoutError:
                        logging.warning("Timeout waiting for tasks to cancel")
                    except Exception as e:
                        logging.error(f"Error during task cleanup: {e}")
            
            finally:
                self.loop.close()

    def start_crawling(self, seed_urls: List[str]):
        """Start the crawler in a separate thread"""
        logging.info(f"Starting crawler with {len(seed_urls)} seed URLs")
        self.thread_pool = threading.Thread(target=self.run_async_loop, args=(seed_urls,))
        self.thread_pool.start()

    def stop(self):
        """Stop the crawler gracefully"""
        logging.info("Stopping crawler...")
        self.is_running = False
        
        if self.thread_pool and self.thread_pool.is_alive():
            try:
                self.thread_pool.join(timeout=10)  # Wait up to 10 seconds
                if self.thread_pool.is_alive():
                    logging.warning("Crawler thread did not stop gracefully within timeout")
            except Exception as e:
                logging.error(f"Error stopping crawler thread: {e}")
        
        logging.info("Crawler stopped")

    def update_domain_stats(self, domain, success=True):
        """Update domain statistics for adaptive rate limiting"""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {'errors': 0, 'success': 0}
        
        if success:
            self.domain_stats[domain]['success'] += 1
            self.metrics['successful_requests'] += 1
        else:
            self.domain_stats[domain]['errors'] += 1
            self.metrics['failed_requests'] += 1

    def should_crawl_url(self, url):
        """Additional URL filtering"""
        parsed = urlparse(url)
        # Skip social media, ads, tracking, etc.
        skip_domains = {'facebook.com', 'twitter.com', 'instagram.com', 'ads.', 'analytics.', 'tracker.'}
        return not any(domain in parsed.netloc for domain in skip_domains)

    def delete_cached_page(self, url):
        """Delete a page from cache after processing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('DELETE FROM pages WHERE url = ?', (url,))
                conn.commit()
        except Exception as e:
            logging.error(f"Error deleting cached page {url}: {e}")

    def delete_visited_url(self, url):
        """Delete a visited URL from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('DELETE FROM visited_urls WHERE url = ?', (url,))
                conn.commit()
        except Exception as e:
            logging.error(f"Error deleting visited URL {url}: {e}")

    def process_cached_content(self, url, content, depth):
        """Process content from cache and delete after processing"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = self.extract_text(soup)
            if text and len(text.split()) > 50:
                if not self.text_queue.full():
                    self.text_queue.put({'url': url, 'text': text, 'depth': depth})
            
            # Delete the processed page from cache
            self.delete_cached_page(url)
            
        except Exception as e:
            logging.error(f"Error processing cached content from {url}: {e}")

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including cache and error stats"""
        metrics = self.get_metrics()
        metrics.update({
            'cache_hits': self.metrics['cached_hits'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (self.metrics['successful_requests'] / 
                           (self.metrics['successful_requests'] + self.metrics['failed_requests'])
                           if self.metrics['successful_requests'] + self.metrics['failed_requests'] > 0 else 0),
            'domain_stats': self.domain_stats
        })
        return metrics

    def get_domain_lock(self, domain):
        with self.domain_lock_lock:
            if domain not in self.domain_locks:
                self.domain_locks[domain] = asyncio.Lock()
            return self.domain_locks[domain]

    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def is_valid_image_url(url: str) -> bool:
        return any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])

    @staticmethod
    def is_valid_video_url(url: str) -> bool:
        video_platforms = ['youtube.com', 'vimeo.com', 'dailymotion.com']
        return any(platform in url.lower() for platform in video_platforms)

    @staticmethod
    def extract_text(soup: BeautifulSoup) -> str:
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        return text

    def get_metrics(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.metrics['start_time']
        return {
            'pages_crawled': self.metrics['pages_crawled'],
            'bytes_downloaded': self.metrics['bytes_downloaded'],
            'elapsed_time': elapsed_time,
            'crawl_rate': self.metrics['pages_crawled'] / elapsed_time if elapsed_time > 0 else 0,
            'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        } 